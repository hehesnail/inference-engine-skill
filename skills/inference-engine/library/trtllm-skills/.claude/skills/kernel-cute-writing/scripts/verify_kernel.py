#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Verify a CuTe DSL kernel against its reference using the fixed-name contract.

Standalone script -- only Python stdlib required (torch/triton needed at
runtime for GPU verification, but not for --mock mode).
Outputs structured JSON to stdout.

Contract:
    The kernel file must export:
    - ``kernel_fn``: callable -- the CuTe DSL kernel wrapper
    - ``reference_fn``: callable -- reference implementation (same signature)
    - ``get_inputs()``: returns a list of CUDA tensors

Usage:
    python verify_kernel.py kernel.py [--rtol 1e-3] [--atol 1e-3] \
        [--timeout 60] [--mock]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap

# ---------------------------------------------------------------------------
# Verification harness generation
# ---------------------------------------------------------------------------


def _build_verification_script(
    kernel_path: str,
    rtol: float,
    atol: float,
) -> str:
    """Generate a temporary verification harness script.

    The harness:
    1. Imports the kernel module using the fixed-name contract
    2. Calls ``get_inputs()`` for shared test tensors
    3. Runs ``reference_fn`` and ``kernel_fn`` with the same inputs
    4. Recursively compares outputs (handles tuples, lists, dicts, tensors,
       scalars)
    5. Prints a structured ``RESULT:`` line to stdout
    """
    abs_kernel_path = os.path.abspath(kernel_path)
    kernel_dir = os.path.dirname(abs_kernel_path)

    return textwrap.dedent(f"""\
        import sys
        import importlib.util
        import math

        sys.path.insert(0, {kernel_dir!r})

        # Import the kernel module
        _spec = importlib.util.spec_from_file_location(
            "kernel_module", {abs_kernel_path!r}
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)

        # Validate fixed-name contract
        for _attr in ("kernel_fn", "reference_fn", "get_inputs"):
            if not hasattr(_mod, _attr):
                print(f"ERROR:kernel file missing required export: {{_attr}}")
                sys.exit(1)

        import torch

        # Get shared inputs
        inputs = _mod.get_inputs()
        if not isinstance(inputs, (list, tuple)):
            print("ERROR:get_inputs() must return a list or tuple of tensors")
            sys.exit(1)

        # Clone inputs for each call to avoid in-place mutation issues
        def _clone(x):
            if isinstance(x, torch.Tensor):
                return x.clone()
            if isinstance(x, dict):
                return {{k: _clone(v) for k, v in x.items()}}
            if isinstance(x, list):
                return [_clone(v) for v in x]
            if isinstance(x, tuple):
                return tuple(_clone(v) for v in x)
            return x

        ref_inputs = [_clone(t) for t in inputs]
        kern_inputs = [_clone(t) for t in inputs]

        # Run both implementations
        ref_out = _mod.reference_fn(*ref_inputs)
        kern_out = _mod.kernel_fn(*kern_inputs)

        # Recursive comparison
        _global_max_abs = 0.0
        _global_max_rel = 0.0
        _all_correct = True
        _mismatches = []


        def _compare(ref, kern, path="root"):
            global _global_max_abs, _global_max_rel, _all_correct
            def fail(reason):
                global _all_correct
                _all_correct = False
                _mismatches.append(f"{{path}}: {{reason}}")

            if isinstance(ref, torch.Tensor) and isinstance(kern, torch.Tensor):
                if (ref.shape != kern.shape or ref.dtype != kern.dtype
                        or ref.device != kern.device or ref.layout != kern.layout):
                    fail("tensor shape/dtype/device/layout differs")
                    return
                if ref.numel() == 0:
                    return
                if not (ref.is_floating_point() or ref.is_complex()):
                    if not torch.equal(ref, kern):
                        fail("integer/bool tensors must match exactly")
                    return
                try:
                    torch.testing.assert_close(
                        kern, ref, rtol={rtol}, atol={atol}, equal_nan=False,
                        check_dtype=True, check_device=True, check_layout=True,
                    )
                except AssertionError as exc:
                    fail(str(exc))
                # Diagnostics must not downcast FP64 or discard imaginary parts.
                diagnostic_dtype = torch.complex128 if ref.is_complex() else torch.float64
                r = ref.to(diagnostic_dtype)
                k = kern.to(diagnostic_dtype)
                diff = (k - r).abs()
                max_abs = diff.max().item()
                magnitude = r.abs()
                safe = torch.where(magnitude > 0, magnitude, torch.ones_like(magnitude))
                max_rel = (diff / safe).max().item()
                if not math.isfinite(max_abs):
                    max_abs = float("inf")
                if not math.isfinite(max_rel):
                    max_rel = float("inf")
                _global_max_abs = max(_global_max_abs, max_abs)
                _global_max_rel = max(_global_max_rel, max_rel)
            elif isinstance(ref, dict) and isinstance(kern, dict):
                if ref.keys() != kern.keys():
                    fail("dictionary keys differ")
                    return
                for key in ref:
                    _compare(ref[key], kern[key], path + f"[{{key!r}}]")
            elif isinstance(ref, (list, tuple)) and isinstance(kern, (list, tuple)):
                if type(ref) is not type(kern) or len(ref) != len(kern):
                    fail("sequence type or length differs")
                    return
                for i, (r, k) in enumerate(zip(ref, kern)):
                    _compare(r, k, path + f"[{{i}}]")
            elif type(ref) is float and type(kern) is float:
                if math.isnan(ref) or math.isnan(kern):
                    fail("NaN scalar")
                elif not (math.isfinite(ref) and math.isfinite(kern)):
                    if ref != kern:
                        fail("non-finite scalar mismatch")
                else:
                    abs_d = abs(kern - ref)
                    rel_d = abs_d / abs(ref) if ref else (0.0 if abs_d == 0 else float("inf"))
                    _global_max_abs = max(_global_max_abs, abs_d)
                    _global_max_rel = max(_global_max_rel, rel_d)
                    if abs_d > {atol} + {rtol} * abs(ref):
                        fail("scalar value differs")
            elif type(ref) is not type(kern) or ref != kern:
                fail("value or type differs")


        try:
            _compare(ref_out, kern_out)
            _result_parts = [
                f"passed={{_all_correct}}",
                f"max_abs={{_global_max_abs}}",
                f"max_rel={{_global_max_rel}}",
            ]
            print("RESULT:" + ",".join(_result_parts))
            for _m in _mismatches:
                print(f"MISMATCH:{{_m}}", file=sys.stderr)
        except Exception as e:
            print(f"ERROR:{{e}}")
            sys.exit(1)
    """)


# ---------------------------------------------------------------------------
# Core verification function
# ---------------------------------------------------------------------------


def verify_kernel(
    kernel_path: str,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    timeout: int = 60,
) -> dict:
    """Verify kernel correctness using the fixed-name contract.

    Args:
        kernel_path: Path to Python file exporting ``kernel_fn``,
            ``reference_fn``, and ``get_inputs()``.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        timeout: Execution timeout in seconds.

    Returns:
        Dict with keys ``correct``, ``max_abs_diff``, ``max_rel_diff``,
        ``details``.
    """
    if not os.path.exists(kernel_path):
        print(f"Kernel file not found: {kernel_path}", file=sys.stderr)
        sys.exit(1)

    script = _build_verification_script(kernel_path, rtol, atol)

    working_dir = os.path.dirname(os.path.abspath(kernel_path))

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=working_dir
    ) as script_file:
        script_file.write(script)
        script_path = script_file.name

    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=working_dir,
        )

        output = proc.stdout + proc.stderr

        if proc.returncode != 0:
            return {
                "correct": False,
                "max_abs_diff": float("inf"),
                "max_rel_diff": float("inf"),
                "details": f"Verification process exited with code {proc.returncode}: {output[:500]}",
            }
        result_lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT:")]
        if len(result_lines) != 1:
            return {
                "correct": False,
                "max_abs_diff": float("inf"),
                "max_rel_diff": float("inf"),
                "details": f"Expected one harness RESULT record, got {len(result_lines)}: {output[:500]}",
            }

        if "RESULT:" in output:
            try:
                result_line = result_lines[0]
                parts_str = result_line.split("RESULT:")[1]
                parts = parts_str.split(",")
                if len(parts) != 3 or parts[0] not in ("passed=True", "passed=False"):
                    raise ValueError("Malformed RESULT status")
                if not parts[1].startswith("max_abs=") or not parts[2].startswith("max_rel="):
                    raise ValueError("Malformed RESULT metrics")
                passed = parts[0] == "passed=True"
                max_abs = float(parts[1].split("=")[1])
                max_rel = float(parts[2].split("=")[1])
            except (IndexError, ValueError) as exc:
                return {
                    "correct": False,
                    "max_abs_diff": float("inf"),
                    "max_rel_diff": float("inf"),
                    "details": (f"Failed to parse RESULT line: {exc}. Raw output: {output[:500]}"),
                }

            if passed:
                details = f"All outputs match within tolerance (rtol={rtol}, atol={atol})"
            else:
                details = (
                    f"Outputs differ beyond tolerance"
                    f" (max_abs={max_abs:.2e}, rtol={rtol}, atol={atol})"
                )

            return {
                "correct": passed,
                "max_abs_diff": max_abs,
                "max_rel_diff": max_rel,
                "details": details,
            }
        elif "ERROR:" in output:
            error_msg = output.split("ERROR:")[1].strip().split("\n")[0]
            return {
                "correct": False,
                "max_abs_diff": float("inf"),
                "max_rel_diff": float("inf"),
                "details": f"Verification error: {error_msg}",
            }
        else:
            return {
                "correct": False,
                "max_abs_diff": float("inf"),
                "max_rel_diff": float("inf"),
                "details": f"Unexpected output: {output[:500]}",
            }

    except subprocess.TimeoutExpired:
        return {
            "correct": False,
            "max_abs_diff": float("inf"),
            "max_rel_diff": float("inf"),
            "details": f"Verification timed out after {timeout} seconds",
        }
    finally:
        if os.path.exists(script_path):
            os.unlink(script_path)


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------


def _mock_data(rtol: float = 1e-3, atol: float = 1e-3) -> dict:
    """Return realistic mock verification data for testing."""
    return {
        "correct": True,
        "max_abs_diff": 1.2e-7,
        "max_rel_diff": 3.4e-6,
        "details": f"All outputs match within tolerance (rtol={rtol}, atol={atol})",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for CLI invocation."""
    parser = argparse.ArgumentParser(
        description="Verify CuTe DSL kernel correctness using fixed-name contract."
    )
    parser.add_argument(
        "kernel_path",
        nargs="?",
        help="Path to Python file exporting kernel_fn, reference_fn, get_inputs().",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance (default: 1e-3).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance (default: 1e-3).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Execution timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Return mock data for testing (no GPU required).",
    )
    args = parser.parse_args()

    if args.mock:
        data = _mock_data(rtol=args.rtol, atol=args.atol)
    elif args.kernel_path:
        data = verify_kernel(
            kernel_path=args.kernel_path,
            rtol=args.rtol,
            atol=args.atol,
            timeout=args.timeout,
        )
    else:
        parser.error("Either --mock or kernel_path is required.")

    json.dump(data, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
