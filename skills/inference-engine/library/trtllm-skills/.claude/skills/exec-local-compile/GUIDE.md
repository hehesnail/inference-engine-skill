> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[NVIDIA/TensorRT-LLM@f405e10b730e](https://github.com/NVIDIA/TensorRT-LLM/blob/f405e10b730ea953961ecf5719242c581527587a/.claude/skills/exec-local-compile/SKILL.md)，2026-10-31 核对。

# Compile TensorRT-LLM (Local / Compute Node)

Compile TensorRT-LLM from source on a compute node inside a Docker container.

## When to Use

| Scenario | Use This Skill? |
|----------|----------------|
| Target CUDA/compiler/TensorRT dependencies and target architecture are known | CPU compilation may be possible; verify build-script probes |
| Login/frontend node | Follow its compute policy; use a permitted CPU build node or Slurm job |

## Prerequisites

- You are inside a Docker/enroot container on a compute node
- Target architectures are explicit; GPU visibility is needed only for GPU-dependent probes or tests
- `/usr/local/tensorrt` exists (TensorRT installation in the container)

## Instructions

### Step 1: Verify Environment

Confirm the permitted build host, compiler/CUDA/TensorRT/Python ABI and target architectures. Compilation alone does not authorize GPU use.

### Step 2: Locate the Codebase

`cd` to the TensorRT-LLM repository. If the path is not provided by the user, ask for it.

### Step 3: (Optional) Checkout Branch

If the user specifies a branch (e.g., "compile ToT"), checkout and pull:
```bash
git status --short
git rev-parse HEAD
# Build the agreed checkout; use an isolated worktree for another revision.
```

### Step 4: Build

Run the build command (**incremental by default** — omit `-c`/`--clean` unless explicitly requested or the incremental build fails):

```bash
./scripts/build_wheel.py --use_ccache -a "<arch>" --nvtx
```

Replace `<arch>` with the target GPU architecture (see Architecture Reference below). If not specified by the user, auto-detect from `nvidia-smi`.

### Step 5: Install

```bash
<isolated_python> -m pip install -e ".[devel]"
# Verify the loaded module path and built extension ABI, not only __version__.
```

### Step 6: Verify

```bash
python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
```

## Build Flags

| Flag | Description |
|------|-------------|
| `-a "<arch>"` | Target GPU architecture(s) |
| `--nvtx` | Enable NVTX markers for profiling |
| `--use_ccache` | Use ccache for faster recompilation |
| `-f` / `--fast_build` | Skip some kernels for faster dev compilation. Use only when the omitted kernels are outside the task coverage; record the reduced build scope. |
| `-c` / `--clean` | Clean build directory before building. Only when needed (see below). |
| `--skip_building_wheel` | Build in-place without creating a wheel file |
| `--no-venv` | Skip virtual environment creation |

## Architecture Reference

| Value | GPU Family |
|-------|-----------|
| `"103"` | Blackwell Ultra (B300, GB300) |
| `"100"` | Blackwell (B200, GB200) |
| `"90"` | Hopper (H100, H200) |
| `"89"` | Ada Lovelace (L40S) |
| `"80"` | Ampere (A100) |
| `"90;100"` | Multiple architectures |

## Incremental vs. Clean Builds

**Default to incremental builds** — CMake only recompiles changed files, saving significant time.

Use a **clean build** (`-c`) only when:
- The user explicitly requests a clean/fresh build
- An incremental build fails with linker errors, stale object files, or CMake cache issues
- Major branch changes (e.g., rebasing across many commits) that may invalidate the build cache
- Build system files changed (`CMakeLists.txt`, `*.cmake`)
