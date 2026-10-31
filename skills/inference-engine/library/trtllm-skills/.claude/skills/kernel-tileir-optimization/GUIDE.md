> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[NVIDIA/TensorRT-LLM@f405e10b730e](https://github.com/NVIDIA/TensorRT-LLM/blob/f405e10b730ea953961ecf5719242c581527587a/.claude/skills/kernel-tileir-optimization/SKILL.md)，2026-10-31 核对。

# Triton TileIR Optimization

Optimize EXISTING Triton kernels for NVIDIA's TileIR backend on Blackwell GPUs.
This skill does NOT write kernels from scratch -- that is the Triton Specialist's job.

## Principles

### TileIR vs PTX Backend

TileIR is NVIDIA's compiler backend for Triton that generates optimized CUDA code
using CGA-level (Cooperative Grid Array) tile representations. Critical differences:

| Parameter | PTX Backend | TileIR Backend |
|-----------|-------------|----------------|
| `num_warps` | Strict directive | **Ignored** (compiler decides) |
| `num_stages` | Strict directive | Cost hint (compiler optimizes) |
| `occupancy` | Not available | **Critical** tuning param (1-32) |
| `num_ctas` | Limited | 2CTA mode for Blackwell |
| Block sizes | Smaller often better | Larger often better |
| TMA | Supported on appropriate NVIDIA targets via tensor descriptors | Verify descriptor, target and backend-version restrictions |

**Key implication**: Do not tune `num_warps` for TileIR -- focus on `occupancy` instead.

### Triton Package Landscape

Three packages share `import triton`:

| Package | Source | Use Case |
|---------|--------|----------|
| `pytorch-triton` | PyTorch wheel | `torch.compile`, standard kernels |
| `triton` | OpenAI PyPI | Official Triton from triton-lang.org |
| nvtriton | [Triton-to-tile-IR](https://github.com/triton-lang/Triton-to-tile-IR) | TileIR backend for Blackwell |

Only one triton package should be installed at a time. "Converting to TileIR" means
adding TileIR-specific configs, NOT changing imports. TileIR activates via `ENABLE_TILE=1`.

### When TileIR Applies

TileIR requires a compatible installed backend, target architecture and compiler/runtime version. Select TileIR configurations only when that backend is active. Standard `triton.Config` does not accept an `occupancy` keyword; it may raise TypeError instead of ignoring it. An SM major-version check alone is insufficient.

**Expected speedups** (with nvtriton on Blackwell):

These upstream ranges have no reproducible workload/version evidence here; treat them as unverified historical examples, not expected gains. Measure the current workload.

| Kernel Type | Unverified upstream range | Candidate Lever |
|-------------|---------|-----------|
| Dot-Related (GEMM, Attention) | 1.2-2.0x | TMA + 2CTA |
| Norm-Like (LayerNorm, Softmax) | 2.0-5.0x | High occupancy |
| Element-Wise (ReLU, Add, Exp) | 1.5-3.0x | Occupancy + num_stages |
| Reduction (Sum, Mean, Max) | 1.8-4.0x | High occupancy |

## Workflow

Five-phase workflow: compatibility, classify, transform, validate, benchmark.

### Phase 1: Compatibility Test (ENABLE_TILE=0)

Verify the kernel works in PTX mode before applying TileIR optimizations.

```bash
python scripts/tileir_check.py
```

Then use the **kernel-triton-writing** skill's `verify_kernel.py` to verify with `ENABLE_TILE=0`:

```bash
ENABLE_TILE=0 python <kernel-triton-writing-dir>/scripts/verify_kernel.py path/to/kernel.py
# kernel.py exports get_inputs(), reference_fn(*inputs), kernel_fn(*inputs).
```

### Phase 2: Classify Kernel

Determine kernel type to select the optimization strategy.

```bash
python scripts/classify_kernel.py --file kernel.py
```

Classification decision tree:

```
Contains tl.dot()?
  YES --> dot-related: TMA + 2CTA + occupancy + larger blocks
  NO  --> Has reduction + normalization?
            YES --> norm-like: backend-supported occupancy candidates
            NO  --> Point-wise only?
                      YES --> element-wise: occupancy (1-16) + num_stages (2-4)
                      NO  --> reduction: backend-supported occupancy candidates
```

### Phase 3: Apply Transformations

Classify and apply optimizations in one step:

```bash
python scripts/classify_kernel.py --file kernel.py --apply-optimizations
```

Output JSON includes `optimized_code` and `changes_applied` fields.

**Type-specific transformations:**

**Dot-related** (highest priority):
1. Convert `tl.load`/`tl.store` to TMA descriptors (MANDATORY). See `references/tma-conversion.md`.
2. Add 2CTA configs (`num_ctas=2`) with SM oversubscription guard in pre-hook.
3. Add occupancy (1, 2, 4) and extended num_stages (4, 6).
4. Use larger block sizes (256x256, 256x128).

**Norm-like** (LayerNorm, Softmax, RMSNorm):
- Explore occupancy values supported by this backend. Do not tune ignored `num_warps` settings; validate the actual lowering and workload.

**Element-wise** (ReLU, GELU, Add, Mul, Exp):
- Add occupancy (1, 2, 4, 16), num_stages (2, 3, 4). Include extreme configs for small inputs.

**Reduction** (Sum, Mean, Max):
- Use the same backend-aware strategy as norm-like kernels; reductions need a verified accumulation/identity contract.

Gate configurations using the active backend and its supported architecture/API contract. Construct backend-specific Config objects only inside that branch:

```python
import triton

def get_configs_with_gating(pre_hook=None):
    target = triton.runtime.driver.active.get_current_target()
    if target.backend == "tileir":
        # Verify this installed backend supports the target and proposed options.
        return get_tileir_specific_configs(pre_hook)
    return get_baseline_configs()
```

See `references/config-templates.md` for complete config templates per kernel type.

The classifier uses text patterns, including comments, and its confidence is heuristic. Generated configs may contain nonexistent parameters such as `n_elements`; review and compile the draft before use. `tileir_compatible: null` means unverified.

### Phase 4: TileIR Validation (ENABLE_TILE=1)

Use the **kernel-triton-writing** skill's `verify_kernel.py` to verify the optimized kernel with TileIR backend:

```bash
ENABLE_TILE=1 python <kernel-triton-writing-dir>/scripts/verify_kernel.py path/to/optimized_kernel.py
```

Set `ENABLE_TILE=1` before running. Check: numerical correctness, no compilation errors,
TMA/2CTA patterns compile successfully.

### Phase 5: Benchmark

Use `triton.testing.do_bench()` (as documented in the **perf-workload-profiling** skill) to compare PTX (`ENABLE_TILE=0`) vs TileIR (`ENABLE_TILE=1`).

Benchmark across multiple input sizes (128, 1024, 8192) -- performance varies by size.

## Scripts

### tileir_check.py

Check TileIR availability (nvtriton, ENABLE_TILE, Blackwell GPU):

```bash
python scripts/tileir_check.py
```

Returns JSON: `nvtriton_installed`, `tileir_active`, `blackwell_gpu`, `gpu_capability`, `recommendation`.

### classify_kernel.py

Classify kernel type and optionally apply TileIR optimizations:

```bash
# Classify only
python scripts/classify_kernel.py --file kernel.py

# Classify + apply optimizations
python scripts/classify_kernel.py --file kernel.py --apply-optimizations

# From inline code
python scripts/classify_kernel.py --code '<kernel_code>'
```

Returns JSON: `classification`, `confidence`, `indicators`, `recommendations`.
With `--apply-optimizations`: adds `optimized_code` and `changes_applied`.

## Error Handling

### Common Pitfalls

**TMA descriptor errors** (dot-related kernels):
- Always pass `pre_hook=tma_set_block_size_hook` to config generation -- without it,
  TMA descriptors keep dummy block sizes, causing runtime errors or wrong results.
- For GEMM: pass `b.T.contiguous()` in wrapper and use `tl.dot(a, b.T, accumulator)`
  in kernel. Transposition mismatch produces incorrect results silently.

**2CTA oversubscription**:
- Adjust SM count in pre-hook when using `num_ctas=2`:
  ```python
  if "NUM_SMS" in nargs and "NUM_CTAS" in nargs:
      nargs["NUM_SMS"] = nargs["NUM_SMS"] // nargs["NUM_CTAS"]
  ```

**Config function signatures**:
- ALL config helper functions MUST accept `pre_hook=None`, even if unused.
  Without it: `TypeError: get_autotune_configs() takes 0 positional arguments`.

**Hardware gating**:
- Gate TileIR configs with `torch.cuda.get_device_capability()[0] >= 10`.
  TMA/2CTA on pre-Blackwell GPUs causes runtime crashes.

**API availability**:
- Use `1.0 / (1.0 + tl.exp(-x))` instead of `tl.sigmoid(x)` -- not available in
  all Triton versions including some nvtriton builds.

**Performance tuning**:
- Do not over-tune `num_warps` -- TileIR ignores it. Focus on `occupancy`.
- Use larger block sizes (256x256, 256x128) for TileIR, not PTX-tuned small blocks.
- Benchmark across small/medium/large inputs; one-size configs underperform.
- Approximate math and flush-to-zero alter the numerical contract. Enable only when the task permits those semantics and validation covers extreme/subnormal inputs:
  ```bash
  export TILEIR_ENABLE_APPROX=1
  export TILEIR_ENABLE_FTZ=1
  ```

### When to Abort

Stop and report if:

1. **No triton installed** -- cannot proceed.
2. **Compatibility test fails** -- kernel has syntax/runtime errors before optimization.
3. **TileIR validation fails** -- optimized kernel produces wrong results.
4. **No speedup** -- TileIR version is slower than PTX baseline (with nvtriton).
5. **Not Blackwell GPU** -- still add configs for future deployment, but skip
   ENABLE_TILE testing and benchmarking.

### Output Format

After optimization, return:

```
## TileIR Optimization: kernel_name

### Classification
- Kernel type: [dot-related | norm-like | element-wise | reduction]
- Strategy: [TMA + 2CTA | High occupancy | Occupancy + num_stages]

### Compatibility Check (ENABLE_TILE=0)
[PASSED | FAILED] — Max difference: X.Xe-Y

### Transformations Applied
- [List of transformations]

### TileIR Validation (ENABLE_TILE=1)
[PASSED | FAILED] — Max difference: X.Xe-Y

### Benchmark Comparison
| Backend | Time (ms) | Speedup |
|---------|-----------|---------|
| PTX (ENABLE_TILE=0) | X.XXX | 1.0x |
| TileIR (ENABLE_TILE=1) | X.XXX | Y.Yx |

### Output
File: kernel_name_tileir.py
```

Primary contracts for the corrections above (verify installed versions): [official source 1](https://triton-lang.org/main/python-api/generated/triton.Config.html), [official source 2](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html).
