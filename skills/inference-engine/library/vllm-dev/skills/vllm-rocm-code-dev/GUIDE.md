> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[shen-shanshan/vllm-dev-skills@1ccafeebf71b](https://github.com/shen-shanshan/vllm-dev-skills/blob/1ccafeebf71b5d922f351a0cd009a258b7474fac/skills/vllm-rocm-code-dev/SKILL.md)，2026-10-31 核对。

# vLLM ROCm Code Dev

Turn a user's development requirement into a ROCm-only code change in vLLM, plus a PR description that stays in sync with the code.

## Paths

- vLLM repo: the current working tree (usually `/home/shashen/vllm/vllm`); confirm with `git rev-parse --show-toplevel`.
- PR descriptions: `/home/shashen/vllm/vllm-dev-skills/docs/tasks/<task-slug>.md`. The `<task-slug>` is short snake_case (e.g. `dsv4_rocm_mxfp4_indexer_default_on`). Record the chosen path at the start of the task and reuse it for all later updates.

## Hard Rules

1. **Confine the change to ROCm.** CUDA, XPU, CPU, TPU and out-of-tree platforms must behave exactly as before.
   - Prefer ROCm-only places first: `vllm/platforms/rocm.py`, `vllm/_aiter_ops.py`, ROCm attention backends, `forward_hip`, `VLLM_ROCM_*` envs, ROCm-only kernels.
   - If shared code must change, add a new hook on the base class (e.g. a `Platform` classmethod) whose default keeps the current behavior exactly, and override it in the ROCm subclass. Call the hook from the shared code. Avoid sprinkling `if current_platform.is_rocm():` through shared code; use it only when a hook would be clearly over-engineered (a one-line guard).
   - Do not change defaults, signatures or semantics of shared APIs. Gate the new ROCm path by arch (`on_gfx950()`, `get_cdna_version()`), aiter availability (`rocm_aiter_ops.is_*_enabled()`) and/or env vars, and fall back silently to the existing path when the conditions are not met.
   - See [references/rocm-isolation-patterns.md](references/rocm-isolation-patterns.md) for extension points and examples.
2. **Comment key new code concisely.** One or two lines explaining *why* this is needed/different on ROCm (hardware, kernel, aiter, numerics constraint) — not what the code does. No long paragraphs, no change-log style comments.
3. **Preserve provenance.** Describe the technique and cite the originating implementation when relevant; retain required copyright/license attribution. Do not hide another engine as the source.
4. **Plan in Chinese.** When in plan mode, write the plan file in Chinese (code identifiers, paths and commands stay as-is).
5. **Keep the PR description in sync.** After the initial implementation and after *every* later code update (rebase on main, bugfix, review fix, refactor, new benchmark data), update the same `docs/tasks/<task-slug>.md`.

## Workflow

### 1. Understand the requirement

- Read the requirement, linked PRs/issues and the relevant vLLM code on the current branch. Check `git status` / `git log -1` so you know the base.
- Find how the feature currently works on CUDA and on ROCm, and where the code paths diverge.
- Ask the user only for things that are genuinely ambiguous (target arch, model, accuracy/perf target).

### 2. Design (plan mode → Chinese plan)

The plan must include:

- 需求理解与目标；
- 改动方案：每个文件改什么，为什么放在这里；
- **对其他后端的影响分析**：逐条说明 CUDA / XPU / CPU 等为何不受影响（新接口的默认实现保持原行为、仅 ROCm 子类覆写、门控条件等）；
- 生效条件与回退路径（arch / aiter / env / model）；
- 验证方案（单测、精度如 GSM8K/lm_eval、性能如 `vllm bench serve`）；
- PR 描述文件路径 `docs/tasks/<task-slug>.md`。

### 3. Implement

- Follow the hard rules above and the surrounding code style (naming, logging via `logger.info_once`/`warning_once`, lazy imports of aiter).
- Keep the diff minimal: no drive-by refactors or unrelated formatting.
- Add or reuse the necessary tests that exercise the changed ROCm path and relevant compatibility boundaries.

### 4. Self-check

Run from the vLLM repo root:

```bash
git diff --stat                          # every touched file is justified in the plan
git diff | grep -in "atom"               # must print nothing
pre-commit run --files $(git diff --name-only)   # preserve failure status; report missing dependencies separately
```

Then re-read the diff of every non-ROCm file and confirm the default/CUDA behavior is byte-for-byte equivalent (same defaults, same branches taken). State the result of this check in the final reply.

### 5. Write the PR description

Invoke the `vllm-pr-desc-generator` skill on the **local diff** (no PR link yet; use `git diff main...HEAD` or the working-tree diff instead of fetching from GitHub). Requirements:

- English, following that skill's format: `**Title:**` with `[ROCm]` scope tag (e.g. `[ROCm][Perf] ...`, `[Bugfix][ROCm] ...`), then Overview / Claims / Validation / Details and the checklist.
- In **Details**, explicitly state why other backends are unaffected (new base-class hook default, ROCm-only override, gating conditions).
- Use real numbers only; put `<TODO: ...>` placeholders for accuracy/perf results not yet measured.
- Save to `/home/shashen/vllm/vllm-dev-skills/docs/tasks/<task-slug>.md` (overwrite only that file).

### 6. Subsequent updates

Whenever the code changes again (rebase on main, bugfix, review comments, new measurements):

1. Make the code change under the same hard rules and rerun the step-4 self-check.
2. Update `docs/tasks/<task-slug>.md` in place: fix affected sections (file list, design, conditions, Validation numbers), remove resolved TODOs. Do not append a change log unless the user asks.
3. Tell the user which sections of the PR description changed.

## Final Reply

Report briefly (in Chinese unless the user writes in English):

- the files changed and the one-line reason for each;
- the other-backend impact check result;
- the PR description path and proposed title;
- remaining `<TODO>`s (e.g. benchmark/accuracy numbers to fill in).
