> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[cohere-ai/vllm-skills@2a9ed1ff5dd8](https://github.com/cohere-ai/vllm-skills/blob/2a9ed1ff5dd822a521a453f49eff84e8dd5a7071/skills/local-test-runner/SKILL.md)，2026-10-31 核对。

# Local Test Runner

## Quick Start
Run vLLM tests locally — either a specific test file or a full Buildkite test area — directly in the current shell with an editable vLLM install.

| Buildkite concept | Local equivalent |
|-------------------|------------------|
| `test_areas/*.yaml` steps | `pytest` commands from those YAML files, run in current env |
| CI working dir `/vllm-workspace/tests` | `<repo_root>/tests` |
| Buildkite sharding (`--num-shards`/`--shard-id`) | Omit locally (run full suite) |
| `$$BUILDKITE_PARALLEL_JOB` vars in YAML | Remove or ignore |

## Workflow

### 1) Ensure environment is set up

In containerized environments, `uv` may fail with `Cross-device link (os error 18)` when its cache and workspace are on different filesystems. If this happens, set the cache to a local directory before any `uv` commands:
```bash
export UV_CACHE_DIR=<repo_root>/.cache/uv
```

Check that vLLM is **fully** installed. An `import vllm` check alone is
insufficient: when Python is run from the repo root, the `vllm/` source
directory shadows the package, so imports can succeed even when
`uv pip install -e .` was never run. In that state there is no `vllm`
console script in `.venv/bin/`, and entrypoint tests that spawn
`vllm serve` via `RemoteOpenAIServer` fail with a confusing
`PermissionError: [Errno 13] Permission denied: 'vllm'` (an empty PATH
entry causes the exec loop to land on the source directory). All three
of these checks must pass:
```bash
.venv/bin/python -c "import vllm; print('vllm version:', vllm.__version__)"
test -x .venv/bin/vllm && echo "vllm CLI OK"
uv pip show vllm > /dev/null 2>&1 && echo "pip metadata OK"
```

If any of the three checks fails, use the `install-vllm` skill to create
the env and install vLLM — do **not** try to patch around a partial
install. Then install test dependencies:
```bash
# Use the target checkout CI dependency file; at vLLM 7b77ed581d92
# select requirements/test/{cuda,rocm,xpu}.in as appropriate.
uv pip install -r requirements/dev.txt
```

### 1b) Prompt for precompiled wheel before running tests

**Always prompt the user for which precompiled wheel to use, unless they have
already specified one.** The precompiled wheel determines which C++/CUDA
binaries are loaded. Using the wrong wheel can cause silent correctness
regressions (e.g. a model producing 100% WER instead of ~12%).

1. Check the currently installed wheel version:
```bash
.venv/bin/python -c "import vllm; print(vllm.__version__)"
```

2. Check the current branch to infer the expected release:
```bash
git rev-parse --abbrev-ref HEAD   # e.g. "v0.19.0-branch" → release 0.19.0
```

3. **Prompt the user** with the currently installed wheel version, the current
   branch name, the inferred matching release version (if any), and these
   options:
     - **Keep current** — proceed with whatever is already installed
     - **Switch to stable release** — reinstall using the release wheel
       matching the branch (via `install-vllm` skill, Option B)
     - **Switch to branch-tip nightly** — reinstall using
       `VLLM_USE_PRECOMPILED=1` (via `install-vllm` skill, Option A)
     - **Use a specific version** — let the user name a version

4. **Wait for the user's answer** before proceeding. Do not assume a default
   and do not run tests until the wheel choice is confirmed.

### 2) Detect GPU (auto)

Run:
```
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

Map common strings: `H100` → `h100`, `H200` → `h200`, `A100` → `a100`, `B200` → `b200`, `GB200` → `gb200`.

Show utilization and suggest which GPU indices appear free. Use only the task-authorized devices. Idle devices are not an allocation; retain explicit CUDA_VISIBLE_DEVICES when provided.

### 3) Determine what to run

Two modes — pick based on user input:

**A) Specific test file or command** (user provides a path or pytest command):
- Accept directly. No need to read Buildkite YAML.
- Example: `tests/v1/core/test_scheduler.py` or `pytest -v -s basic_correctness/test_basic_correctness.py -k test_add_requests`

**B) Buildkite test area** (user names an area):
- `test_area`: one of `attention`, `basic_correctness`, `benchmarks`, `compile`, `cuda`, `distributed`, `e2e_integration`, `engine`, `entrypoints`, `expert_parallelism`, `kernels`, `lm_eval`, `lora`, `misc`, `model_executor`, `model_runner_v2`, `models_basic`, `models_distributed`, `models_language`, `models_multimodal`, `plugins`, `pytorch`, `quantization`, `ray_compat`, `samplers`, `spec_decode`, `weight_loading`
- Read `.buildkite/test_areas/<test_area>.yaml` and extract the `commands:` list. If there are multiple steps (different device requirements or GPU counts), show them and let the user pick.
- Strip Buildkite-specific variables (`$$BUILDKITE_PARALLEL_JOB`, `$$BUILDKITE_PARALLEL_JOB_COUNT`) and `--num-shards` / `--shard-id` flags.
- Replicate any `export` lines (especially `VLLM_WORKER_MULTIPROC_METHOD=spawn`).
- Run any extra `uv pip install` or `pip install` lines from the YAML before the test commands.

Optional parameters for both modes:
- `cuda_visible_devices`: comma-separated GPU indices (defaults to all)
- `num_gpus`: for test areas, read from YAML `num_devices` field (default 1)

### 4) Ensure HF token is available (if needed)

In Buildkite CI, `HF_TOKEN` is pre-set on agents (via secrets/hooks or `/etc/environment`) and passed through to containers with `-e HF_TOKEN`. There is no YAML-level declaration — scripts simply assume it exists in the host environment.

**Skip this step** for unit tests that don't download models (e.g. scheduler, core, engine unit tests).

Only needed when tests load models from Hugging Face (model tests, lm_eval, benchmarks, etc.):
1. Check whether `HF_TOKEN` is set without printing it: `test -n "${HF_TOKEN:-}" && echo "HF_TOKEN is set"`.
2. If set and non-empty, proceed.
3. If not set, check the cached token file (`~/.cache/huggingface/token`). If it exists and is non-empty, export it: `export HF_TOKEN=$(cat ~/.cache/huggingface/token)`
4. If neither is available, ask the user to run `huggingface-cli login` and then retry.
5. **Never echo or log the token value.**

### 5) Run tests

Always use `.venv/bin/python -m pytest` — `source .venv/bin/activate` does not persist across agent shell calls.

**Important:** Always add `.venv/bin` to `PATH`. Some tests (e.g. entrypoint tests using `RemoteOpenAIServer`) spawn `vllm serve` as a subprocess and expect the `vllm` CLI to be on PATH. Without this, those tests fail with `PermissionError: [Errno 13] Permission denied: 'vllm'`.

```bash
set -o pipefail
mkdir -p <repo_root>/.cache/test-logs
cd <repo_root>
export PATH="<repo_root>/.venv/bin:$PATH"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=<cuda_visible_devices>  # omit if using all GPUs
# Only if needed:
export HF_HOME=~/.cache/huggingface
export HF_TOKEN="<hf_token>"

.venv/bin/python -m pytest <test_path_or_commands> -v \
  2>&1 | tee .cache/test-logs/<test_name>-$(date +%Y%m%d-%H%M%S).log
```

Useful pytest flags for local runs:
- `-x` — stop on first failure (add when debugging a specific failure)
- `-k <expression>` — run only tests matching a name pattern
- `-s` — disable output capture (show prints)
- `--timeout=<seconds>` — per-test timeout

### 6) Store logs

Save test output under `<repo_root>/.cache/test-logs/`. Create the directory if it doesn't exist:
```bash
mkdir -p <repo_root>/.cache/test-logs
```

Log file naming: `<test-name>-<YYYYMMDD-HHMMSS>.log`, e.g. `test_scheduler-20260421-175200.log`.

Use `tee` to both display output and write to the log file:
```bash
set -o pipefail
.venv/bin/python -m pytest tests/v1/core/test_scheduler.py -v \
  2>&1 | tee .cache/test-logs/test_scheduler-$(date +%Y%m%d-%H%M%S).log
```

For test area runs, name the log after the area: `basic_correctness-20260421-175200.log`.

After the run, report the log file path to the user.

### 7) Monitoring long-running tests

Tests can take minutes to hours. Run them as background commands (`block_until_ms: 0`) and monitor by polling the terminal output file. Check periodically with exponential backoff until the exit code footer appears.

## Notes
- Buildkite YAML `commands:` use paths relative to `/vllm-workspace/tests`. Locally these are relative to `<repo_root>/tests`.
- Some test areas have `working_dir: "/vllm-workspace/"` — locally that maps to `<repo_root>/`.
- For multi-GPU distributed tests, ensure enough GPUs are available per the YAML `num_devices`.
- If a step fails due to missing dependencies, install and retry.
- The editable install picks up Python source changes immediately; only C++/CUDA extension changes need a reinstall.

## Example prompt flow

### Specific test file
1. User: "run tests in tests/v1/core/test_scheduler.py"
2. Check env is set up (vLLM installed, test deps present)
3. **Verify wheel**: show installed version, branch, and inferred release;
   prompt user to confirm or switch
4. Detect GPUs, check utilization
5. Skip HF token (unit tests don't need it)
6. `mkdir -p .cache/test-logs`
7. Run pytest with log capture under `set -o pipefail` so tee cannot hide test failure.
8. Monitor, report results and log path

### Buildkite test area
1. User: "run the basic_correctness CI tests"
2. Check env, detect GPUs
3. **Verify wheel**: show installed version, branch, and inferred release;
   prompt user to confirm or switch
4. Read `.buildkite/test_areas/basic_correctness.yaml`, show extracted commands
5. Check if HF token is needed (yes for model tests, no for pure unit tests)
6. Run extracted commands with log capture
7. Report results and log path

### User specifies wheel upfront
1. User: "use the v0.19.0 wheel and run tests/entrypoints/test_foo.py"
2. Check env; install with v0.19.0 wheel (skip prompt — user already chose)
3. Detect GPUs, check utilization
4. Run test with log capture
5. Report results and log path