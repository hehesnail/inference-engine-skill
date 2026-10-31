> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[cfregly/gpu-perf-tune@25d74d031732](https://github.com/cfregly/gpu-perf-tune/blob/25d74d031732e2f76d30a43e44c3434b588f06b3/plugins/profile-and-optimize/skills/inference-tune-sweep/SKILL.md)，2026-10-31 核对。

# Inference tune sweep

Run a small, hypothesis-driven serving matrix. Keep one run ID from evidence
capture through the final report. Treat every result as DRAFT until the
production-shaped baseline, repeated trials, and placement receipts support a
verdict.

## Required inputs

Stop if any required input is missing:

1. A stable Ready deployment and a benchmark smoke result.
2. A campaign YAML with explicit deployment, workload, and scheduler settings.
3. One focus metric: throughput, latency, or mixed.
4. A named baseline from the same workload, model, source revision, delivery
   method, hardware, and measurement method.
5. Enough free capacity for the bounded matrix.

For latency tests, set `num_prompts` to at least twice the target concurrency.
Record input length, output length, concurrency, cache state, graph mode, and
`max_num_seqs` for every cell.

## 1. State the hypothesis and bound the grid

Use the cost ledger in
[`performance-hints.md`](https://github.com/cfregly/gpu-perf-tune/blob/25d74d031732e2f76d30a43e44c3434b588f06b3/plugins/profile-and-optimize/server/docs/performance-hints.md) to name the
measured contributor that each axis can change. Drop an axis that has no
plausible connection to the focus metric.

A two by three grid is often enough. Useful axes include concurrency,
`max_num_batched_tokens`, KV cache dtype, graph mode, chunked prefill, and
prefix caching. Change one mechanism at a time when the matrix is meant to
support a causal claim.

## 2. Create one evidence and campaign identity

Call `evidence_init` with the family, intent, and run ID. Use that run ID as the
experiment ID, campaign ID, and `experiment=<run-id>` label for cluster
objects.

Then call `perf_tune_report_campaign_init` with arguments like:

```json
{
  "args": [
    "--config", "<matrix.yaml>",
    "--experiment-id", "<run-id>",
    "--family", "<family>",
    "--evidence-bundle", "<bundle-path>"
  ]
}
```

Record the resulting campaign path in the evidence bundle.

## 3. Dry-run the matrix

Call `perf_tune_report_campaign_run` with the campaign config and `--dry-run`
in `args`. Inspect every rendered deployment change and benchmark command.
Confirm the namespace, node selection, image or source revision, model,
workload shape, warmup, trial count, and cleanup behavior.

Do not hide the job acknowledgement in `args`. The MCP boundary rejects raw
acknowledgement flags.

```json
{
  "args": [
    "--config", "<matrix.yaml>",
    "--campaign", "<campaign-path>",
    "--dry-run"
  ]
}
```

## 4. Record the baseline

Use `perf_baseline_record` before the live sweep. Pass
`--direction higher-is-better` for throughput or
`--direction lower-is-better` for latency. Store the focus metric and
enough structured metadata to prove that later cells are comparable. A
baseline from a different node class, cache state, graph mode, or measurement
method is not a valid comparison.

## 5. Run the live campaign

Call `perf_tune_report_campaign_run` with the same config and the structured
acknowledgement:

```json
{
  "args": [
    "--config", "<matrix.yaml>",
    "--campaign", "<campaign-path>"
  ],
  "i_understand_this_mutates_cluster": true
}
```

This acknowledgement covers the current call's node cordon, Helm release
changes, and benchmark execution. It applies only to the current call. Stop on
a red cell unless the operator explicitly chose continue-on-red behavior.

## 6. Prove placement without private helpers

For a same-node claim, express node selection in the checked-in campaign YAML
or in the scheduler configuration supplied by the operator. Verify the actual
placement after scheduling. For Kubernetes, capture a read-only query such as:

```bash
kubectl get pod <pod> -n <namespace> -o jsonpath='{.spec.nodeName}'
```

Capture that output with `tools/shared/capture_cmd.sh`. Slurm campaigns must
capture the allocated nodes from the scheduler receipt. Run at least three
trials per arm for a verdict. If placement or repeatability cannot be proved,
label the comparison DRAFT.

## 7. Aggregate, compare, and select

Run `perf_tune_report_atlas_aggregate`, then compare the best cells with the
recorded baseline using `perf_baseline_diff`. Use the tolerance declared before
the run.

Call `perf_tune_report_champion_select` only after these checks:

- The focus metric improves beyond the declared tolerance.
- Accuracy and workload coverage meet the campaign requirements.
- Warm and cold results are not mixed.
- Eager and graph-mode results are not mixed.
- Same-node and repeated-trial claims have receipts.

Pass `--same-node` only when the evidence proves it. A within-noise tie keeps
the baseline.

## 8. Render and publish

Run `perf_tune_report_report_render` for the local report. Publishing is strict
by default. Use `--no-strict` only when the operator accepts the reported
evidence gaps, and keep the result DRAFT.

`perf_tune_report_publish_to_lake --dry-run` is local-only. A real external
publish requires the structured field
`i_understand_this_publishes_externally: true` on that call.

## Stop conditions

Stop and report instead of selecting a champion when:

- The baseline is not production-shaped.
- A matrix cell changes an unrecorded variable.
- The benchmark client, workload shape, or cache state differs across arms.
- Capacity forces cross-node comparisons without placement receipts.
- The improvement is within tolerance or accuracy regresses.
- The evidence cannot identify the code and delivery method that ran.

## Outputs

Return the evidence bundle, campaign directory, exact matrix, baseline diff,
placement receipts, trial summaries, chosen config or no-improvement result,
and rendered report. State DRAFT or VERDICT explicitly.

See also:

- [`inference-perf-tune-report`](../inference-perf-tune-report/GUIDE.md)
- [`perf-baseline-record`](../perf-baseline-record/GUIDE.md)
- [`perf-baseline-diff`](../perf-baseline-diff/GUIDE.md)
- [`inference-performance-hints`](../inference-performance-hints/GUIDE.md)
