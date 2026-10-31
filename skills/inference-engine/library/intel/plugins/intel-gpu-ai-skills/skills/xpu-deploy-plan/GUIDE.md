> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[intel/gpu-ai-skills@d0f47fcf2625](https://github.com/intel/gpu-ai-skills/blob/d0f47fcf262560b06fea1684c70d80c877e04854/plugins/intel-gpu-ai-skills/skills/xpu-deploy-plan/SKILL.md)，2026-10-31 核对。

# xpu-deploy-plan

End-to-end deployment **planner** for Intel XPU inference. This skill is
an orchestrator only: it chains the lower-level skills, captures their
outputs, and writes a single `PLAN.md` for the user. It does not
duplicate runtime guidance that lives in the called skills.

Default target: Arc Pro B70 (32 GiB) home-lab host, container-first,
OpenAI-compatible HTTP serving through vLLM-XPU, unless the user's
request points elsewhere.

Before invoking the plan builder, summarize the resolved inputs back to
the user (model, runtime, target XPU, ctx, concurrency, quant) so they
can correct anything before files are written. This is courtesy, not a
safety gate — the script is read/compute only and writes a markdown
plan to `.out/skills/xpu-deploy-plan/`; it does not pull images, launch
containers, claim a GPU, or modify the system.

## When to use

Use when the user asks for a **plan** — wants the orchestration, not
just to start a server:

- "Give me a deployment plan for Qwen2.5-7B on my Arc Pro B70 with vLLM."
- "Plan an end-to-end deployment of Llama-3.1-8B on my Intel GPU."
- "Which skills should I run, and in what order, to serve this model?"
- "I have two/four B70s; give me a launch plan."

## Required entry point

**Always invoke `scripts/build_plan.sh`**. It is the single supported
entry point for this skill. Do not run the chained skills
(`xpu-runtime-preflight`, `model-can-it-fit`, `model-config-recommend`,
or the runtime launch emitters) by hand and assemble a plan yourself —
`build_plan.sh` already orchestrates them, captures their outputs, and
writes the canonical `PLAN.md`. Hand-rolled equivalents drift from the
artifact layout the rest of the toolchain expects and miss the halt
logic on a preflight FAIL.

If `build_plan.sh` fails or a chained skill returns a hard error,
report the failure and the routing skill from the preflight
`SUMMARY.md`; do not work around it by running the underlying scripts
directly.

## Quick start

```sh
plugins/intel-gpu-ai-skills/skills/xpu-deploy-plan/scripts/build_plan.sh \
    --model Qwen/Qwen2.5-7B-Instruct \
    --runtime vllm \
    --target-gpu 0 \
    --ctx 4096 \
    --concurrency 1 \
    --out-dir .out/skills/xpu-deploy-plan
```

The script writes:

```text
.out/skills/xpu-deploy-plan/PLAN.md
.out/skills/xpu-deploy-plan/inputs.json
.out/skills/xpu-deploy-plan/preflight/        # forwarded from xpu-runtime-preflight
.out/skills/xpu-deploy-plan/fit.txt           # from model-can-it-fit
.out/skills/xpu-deploy-plan/config.txt        # from model-config-recommend (vLLM only)
.out/skills/xpu-deploy-plan/build_plan.log
```

`PLAN.md` is the single artifact to show the user. It contains: inputs,
readiness summary, fit summary, the exact launch command from the
selected runtime skill, a smoke test that verifies XPU usage (not just
HTTP 200), the benchmark handoff, and a rollback command.

The output directory `.out/skills/xpu-deploy-plan/` is the
agent-accessible artifact dir for this skill (matches the `.out/` entry
in this repo's `.gitignore`).

## Clarify only what changes the launch

If missing, ask at most three questions; otherwise use defaults.

| Question | Default |
|---|---|
| Which HF model id or local model path? | Required. Do not guess. |
| Runtime: vLLM / SGLang / PyTorch? | vLLM OpenAI API. |
| Target context and concurrency? | 4096 ctx, concurrency 1 for first boot. |
| Quantized weights acceptable? | BF16 if it fits; FP8/int4 only if needed. |
| How many XPUs? | Discover host; do not assume. |

## Chained skills (orchestration only)

The script invokes these skills in order and HALTS on a hard failure:

1. **xpu-runtime-preflight** — readiness gate. If `FAIL` count > 0, the
   plan stops with the preflight `SUMMARY.md` as the blocker. Do not
   attempt launch.
2. **model-can-it-fit** — VRAM sizing for the given model, ctx,
   concurrency, quant, and XPU count.
3. Runtime selection: `vllm` (default), `sglang`, or `torch`.
4. **model-config-recommend** — only for `vllm` on Arc B-series; emits
   recommended flags.
5. Launch command — emitted from the selected runtime skill's
   launch emitter. For vLLM, this is
   `vllm-xpu-run/scripts/emit_launch.sh`, using the top
   `model-config-recommend` candidate when available. **Image policy,
   proxy env propagation, multi-GPU oneCCL setup, and OOM recovery
   live in those skills** (**vllm-xpu-run**, **sglang-xpu-run**,
   **torch-xpu-run**) and the runtime triage skill — they are NOT
   duplicated here.
6. Smoke test — `/v1/models`, one deterministic request, plus
   `docker logs | grep -iE 'xpu|device|cpu fallback'` and `xpu-smi ps`
   to confirm XPU is actually used.
7. Benchmark handoff — points at `vllm-xpu-bench`, `sglang-xpu-bench`,
   or `torch-xpu-bench`.
8. Rollback — `docker stop` / `docker rm` for containers started by the
   plan. The script never removes unrelated containers or caches.

## Runtime decision table

| User need | Runtime | Skill |
|---|---|---|
| OpenAI-compatible chat/completions | vLLM-XPU | **vllm-xpu-run** |
| Maximum current Intel model coverage | vLLM-XPU | **vllm-xpu-run** |
| Prefix-cache-heavy workloads | SGLang-XPU | **sglang-xpu-run** |
| Grammar-constrained generation | SGLang-XPU | **sglang-xpu-run** |
| Pure Transformers code path | PyTorch XPU | **torch-xpu-run** |
| Diffusion / custom HF pipeline | PyTorch XPU | **torch-xpu-run** |
| "How much VRAM?" | Sizing tool | **model-can-it-fit** |
| "Best vLLM config?" | Recommender | **model-config-recommend** |

## Correctness gates

Bringing up the server is not the stopping point. The plan is not
"done" until the smoke test in `PLAN.md` passes and the agent returns
the observed response summary to the user:

- Server exposes the expected model at `/v1/models`.
- One small chat/completion request returns non-empty plausible text;
  quote or summarize the returned content in the final answer.
- Logs show the expected runtime/device path, not CPU fallback.
- XPU memory rises by approximately model-size scale.
- Bench numbers are captured after warmup.

## What this skill does NOT cover

- Driver / `/dev/dri` / group / Docker readiness — **xpu-runtime-preflight**.
- Low-level GPU inventory — **xpu-discover**.
- Detailed vLLM / SGLang / PyTorch flags, image choice, multi-GPU
  oneCCL guidance, and OOM recovery — the selected runtime skill and the
  runtime triage skill own those.
- Performance root-cause analysis — bench / profile skills in order:
  runtime bench, runtime profile, **xpu-profile-unitrace**.

## Reporting

In final answers, report:

- target host and target GPU id
- the readiness verdict (PASS / WARN / FAIL counts from preflight)
- the chosen runtime, image tag, and one-line launch summary
- the path to `PLAN.md` for the full plan
- if `FAIL`: the blocker and the routing skill from preflight `SUMMARY.md`
