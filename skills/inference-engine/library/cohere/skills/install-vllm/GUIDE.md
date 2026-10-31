> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[cohere-ai/vllm-skills@2a9ed1ff5dd8](https://github.com/cohere-ai/vllm-skills/blob/2a9ed1ff5dd822a521a453f49eff84e8dd5a7071/skills/install-vllm/SKILL.md)，2026-10-31 核对。

# Install vLLM (NVIDIA CUDA)

All installs are **editable** (`-e .`) from the current vLLM repo checkout.

## Prerequisites

Verify `uv` is installed before proceeding:

```bash
command -v uv || (echo "uv not found — install via: curl -LsSf https://astral.sh/uv/install.sh | sh" && exit 1)
```

## Step 1: Create and activate the environment

Run from the root of the vLLM repo:

```bash
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
```

## Step 2: Choose the precompiled wheel

**Always prompt the user for which precompiled wheel to use, unless they have
already specified one.** The precompiled wheel determines which C++/CUDA
binaries are loaded at runtime. Using the wrong wheel (e.g. a nightly from the
branch tip when you need the stable release) can cause silent correctness
regressions. The Python source code comes from the editable checkout regardless
of which wheel is chosen.

### 2a) Detect the current branch and infer the default version

```bash
BRANCH=$(git rev-parse --abbrev-ref HEAD)   # e.g. "v0.19.0-branch" or "main"
CPU_ARCH=$(uname -m)                         # x86_64 or aarch64
```

Map branch names to release versions:
- `v<X>.<Y>.<Z>-branch` → release `<X>.<Y>.<Z>` (e.g. `v0.19.0-branch` → `0.19.0`)
- `main` or any other branch → no matching release; default to the branch-tip
  nightly wheel

### 2b) Prompt the user for which wheel to use

Always show the user these options and **wait for their answer** before
proceeding. Do not assume a default — the user must explicitly choose:

1. **Auto-resolved precompiled wheel** — the resolution depends on the target setup.py. At vLLM 7b77ed581d92 it searches backwards from the merge-base with upstream main; it is not guaranteed to match the branch tip. Record the selected wheel commit.
2. **Matching stable release** — the GitHub release wheel for the version that
   matches the branch (if one exists). Uses `VLLM_PRECOMPILED_WHEEL_LOCATION`.
3. **Specific release** — a user-specified version (e.g. "use the v0.18.2
   wheel"). Uses `VLLM_PRECOMPILED_WHEEL_LOCATION`.

Include the detected branch name and inferred version in the prompt so the user
has full context.

### 2c) Install with the chosen wheel

#### Option A: Auto-resolved precompiled wheel

Resolves a compatible published wheel according to the target setup.py and installs editable Python sources. This does not compile modified C++/CUDA code; verify the resolved extension identity.

```bash
VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto
```

If the wheel for the current commit isn't available yet (just merged),
fall back to the previous commit:

```bash
VLLM_PRECOMPILED_WHEEL_COMMIT=$(git rev-parse HEAD~1) \
  VLLM_USE_PRECOMPILED=1 \
  uv pip install --editable . --torch-backend=auto
```

#### Option B: Specific GitHub release wheel

Use for stable releases or when the user names a version. This uses the release
wheel for precompiled binaries but still installs the current repo checkout in
editable mode via `VLLM_PRECOMPILED_WHEEL_LOCATION`.

**Never hand-construct the wheel filename.** The `manylinux` tag and CUDA
variant suffix change between releases (e.g. `manylinux1` → `manylinux_2_31`
→ `manylinux_2_35`, `+cu128` → `+cu130`), so guessing the URL leads to
hard-to-diagnose `HTTP 404` build failures. Instead, always list the release
assets first and pick the exact asset name that matches the target CUDA
variant:

```bash
VLLM_VERSION="0.19.1"         # from user selection or branch inference
CPU_ARCH=$(uname -m)           # x86_64 or aarch64

# List all wheel assets for this release and pick the one you want.
curl -s "https://api.github.com/repos/vllm-project/vllm/releases/tags/v${VLLM_VERSION}" \
  | grep -oE '"browser_download_url": "[^"]+\.whl"' \
  | sed 's/.*"\(.*\)"/\1/' \
  | grep -E "cp38-abi3-manylinux.*${CPU_ARCH}\.whl$"
# Output examples:
#   .../vllm-0.19.1-cp38-abi3-manylinux_2_31_x86_64.whl         ← default CUDA
#   .../vllm-0.19.1+cu130-cp38-abi3-manylinux_2_35_x86_64.whl   ← CUDA 13
#   .../vllm-0.19.1+cpu-cp38-abi3-manylinux_2_35_x86_64.whl     ← CPU-only
```

Pick the **plain** asset (no `+cpu` / `+cuXXX` suffix) for the default CUDA
wheel, or the matching `+cu<version>` asset for a specific CUDA variant.
Copy its URL verbatim into `VLLM_PRECOMPILED_WHEEL_LOCATION`:

```bash
VLLM_PRECOMPILED_WHEEL_LOCATION="<url from the listing above>" \
  uv pip install --editable . --torch-backend=auto
```

For a **specific CUDA variant** that needs a matching PyTorch index, add
`--extra-index-url`:

```bash
CUDA_VERSION="130"
VLLM_PRECOMPILED_WHEEL_LOCATION="<+cu${CUDA_VERSION} url from the listing>" \
  uv pip install --editable . \
  --extra-index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION}"
```

Before running the `uv pip install`, **verify the URL exists** to catch typos
or deleted assets early:

```bash
curl -sfI "$VLLM_PRECOMPILED_WHEEL_LOCATION" > /dev/null \
  && echo "wheel URL OK" \
  || { echo "wheel URL 404: $VLLM_PRECOMPILED_WHEEL_LOCATION"; exit 1; }
```

#### ROCm/CUDA hybrid environments

Some systems have both ROCm (`/opt/rocm`, `rocminfo`) and CUDA installed.
vLLM's `setup.py` auto-detects ROCm first, which causes the precompiled
install to fail looking for a ROCm wheel on what is actually a CUDA machine.

**Diagnosis:** The install fails with
`ValueError: No compatible wheel found for x86_64 at https://pypi.amd.com/vllm-rocm/simple/vllm/`.

**Fix:** Use Option B above with an explicit `VLLM_PRECOMPILED_WHEEL_LOCATION`
to bypass auto-detection entirely.

### 2d) Verify the install (mandatory)

`import vllm` alone is not a sufficient check: the repo's own `vllm/`
directory shadows the package when Python is run from the repo root, so
imports can succeed even when `uv pip install -e .` was never actually
executed. In that case the `vllm` CLI console script is missing from
`.venv/bin/`, which later causes confusing `PermissionError: [Errno 13]
Permission denied: 'vllm'` failures in entrypoint tests (the exec loop
falls back to `./vllm`, which is the source directory). Always run all
four of these checks and fail loudly if any of them fail:

```bash
.venv/bin/python -c "import vllm; print('vllm version:', vllm.__version__)"
test -x .venv/bin/vllm && .venv/bin/vllm --help > /dev/null && echo "vllm CLI OK"
uv pip show vllm > /dev/null 2>&1 && echo "pip metadata OK"
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'torch.cuda.is_available() is False'; print('torch', torch.__version__, 'CUDA OK')"
```

The fourth check matters because `--torch-backend=auto` can silently pick
a CUDA variant newer than the system driver supports — for example
selecting `torch==X.Y.Z+cu130` on a host whose driver advertises max CUDA
12.9. In that case the first three checks pass but the first inference
call dies with `RuntimeError: The NVIDIA driver on your system is too
old (found version 12090)`. If the CUDA check fails, re-run Step 2c with
an explicit `--torch-backend=cu<NN>` matching the `CUDA Version` reported
by `nvidia-smi` (e.g. `--torch-backend=cu128` for driver-reported CUDA
12.9 — torch's cu128 wheels are forward-compatible across 12.x).

If any check fails, re-run Step 2c — do not proceed to later steps or to
test runs with a half-installed environment.

## Step 3: Install pre-commit

```bash
uv pip install -r requirements/lint.txt
pre-commit install
```

## Step 4: Install test dependencies (if running tests)

```bash
# Select the dependency file used by the target checkout/CI backend.
# vLLM 7b77ed581d92 uses requirements/test/{cuda,rocm,xpu}.in;
# requirements/test.in is a historical path, not a current command.
```

## Notes

- **Wheel choice matters for correctness.** The precompiled wheel contains all C++/CUDA kernels. Using a nightly wheel when the stable release is intended (or vice versa) can cause silent correctness regressions — tests may pass or fail depending on which kernels are loaded. Always confirm the wheel choice with the user.
- **Cross-device link errors:** In containerized environments, `uv` may fail with `Cross-device link (os error 18)` when its cache and the workspace are on different filesystems. Fix by setting `export UV_CACHE_DIR=<repo_root>/.cache/uv` before any `uv` commands.
- **Never use system `python3` or bare `pip`.** Always use `uv` and `.venv/bin/python`.
- The default CUDA version for pre-built wheels changes between releases — always check the release assets rather than hardcoding filenames.
- Older releases (< v0.18) use `manylinux1` platform tags and different CUDA variant suffixes (e.g. `+cu121`, `+cu118`).
- `--torch-backend=auto` tells uv to pick the right PyTorch index for your CUDA driver, but it can over-shoot the driver's max CUDA version. If the CUDA check in Step 2d fails, pin explicitly via `--torch-backend=cu<NN>`. `cu128` is a safe default for most current NVIDIA drivers (>= 525).
- If you rebase the dev branch, uninstall vllm and re-run the install to keep precompiled libraries in sync.