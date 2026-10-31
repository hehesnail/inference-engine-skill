> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[cfregly/gpu-perf-tune@25d74d031732](https://github.com/cfregly/gpu-perf-tune/blob/25d74d031732e2f76d30a43e44c3434b588f06b3/plugins/profile-and-optimize/skills/inference-spec-decode-tune/SKILL.md)，2026-10-31 核对。

# Tune speculative decoding draft training

## Scope

This skill defines the experiment contract and gates. It does not ship a draft
trainer, tuning driver, deployment manifest, or serving evaluator. The
operator must supply those commands from the stack under test.

Use it after a single draft has passed capture correctness and stable training.
Use [`inference-spec-decode-train`](../inference-spec-decode-train/GUIDE.md)
first when the recipe itself is not yet proven. Use
[`inference-tune-sweep`](../inference-tune-sweep/GUIDE.md) for serving settings
rather than draft training settings.

## Required inputs

- Exact train, export, serve, and evaluation commands.
- A fixed target checkpoint, corpus hash, tokenizer, and chat template.
- A production-shaped serving workload and standing baseline.
- A small search space with a stated cost ceiling.
- A scheduler capacity check and cleanup plan.
- An acceptance floor plus one user-facing latency objective.

Stop if any candidate would change the checkpoint, corpus, evaluation traffic,
hardware allocation, or serving protocol.

## Workflow

### 1. Bound the search

Choose two to eight candidates. Start with global batch, learning rate,
accumulation, and warmup only when each axis has a plausible mechanism. Keep
the per-device batch within the proven memory limit. State the maximum jobs,
GPU hours, and early-stop rule before submitting anything.

Write a machine-readable matrix in the evidence bundle. Each candidate needs a
stable ID and exact values. Do not call a hand-written grid Bayesian or
Hyperband. Adaptive search requires an operator-provided implementation and a
separate validation of its state and resume behavior.

### 2. Create evidence and preview commands

Call `evidence_init` once for the sweep. For each candidate, render the exact
operator-provided train, export, serve, and evaluation argv. Preview them
without cluster mutation. Verify unique object names and the shared
`experiment=<run-id>` label.

Use the checked-in capture helper for each command from the server root:

```bash
ART_DIR=<bundle> bash tools/shared/capture_cmd.sh <candidate-step> -- \
  <operator-command> <args...>
```

The helper records exact argv, stdout, stderr, and exit code. A nonzero command
stops that candidate.

### 3. Run the training proxy stage

Submit only after the operator approves the rendered jobs. Compare training
accuracy or loss at the same number of consumed samples. Use that proxy only to
drop clearly broken or weak candidates. Preserve the full logs and early-stop
reason for every candidate.

Proxy results are DRAFT. They cannot select the serving champion.

### 4. Evaluate survivors in serving

Export each survivor with the same method and deploy it under an isolated name.
Measure accepted tokens, draft tokens, request errors, throughput, TTFT, and
TPOT or ITL on the same production-shaped workload. Use the same node, graph
mode, warmup, request count, and trial count as the standing config.

When the serving comparison uses a perf report campaign, initialize it with
`--config <matrix.yaml>` and record the returned campaign path. Dry-run
`perf_tune_report_campaign_run` with both `--config` and `--campaign`. For the
live call, pass `i_understand_this_mutates_cluster: true` as a structured MCP
field.

### 5. Apply directional gates

Record immutable scalar baselines with explicit directions:

- Acceptance and throughput use `--direction higher-is-better`.
- TPOT, ITL, and TTFT use `--direction lower-is-better`.

The winner must pass the acceptance floor and improve the declared latency
objective outside the tolerance and observed noise. A proxy winner that loses
the serving comparison is a rejected candidate.

### 6. Report and clean up

Render the campaign report and mark unmatched or single-trial results DRAFT.
Capture candidate IDs, exact settings, training sample counts, serving metrics,
failure reasons, and output hashes. Delete only experiment-labeled resources
after durable outputs are verified.

## Safety rules

- Never modify the standing deployment before a measured win.
- Never reuse a candidate name or output directory across runs.
- Never infer an idle node from a label alone. Check active scheduler and pod
  state before submission.
- Never hide a submit or mutation acknowledgement inside raw MCP args.
- Stop the sweep when the declared cost ceiling is reached.

## Reporting

Every number must include model, checkpoint, corpus hash, hardware,
parallelism, training settings, consumed samples, serving config, workload
shape, concurrency, source revision, trial count, and named baseline. End with
the next candidate mechanism and the measurement that would justify it.
