> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[cfregly/gpu-perf-tune@25d74d031732](https://github.com/cfregly/gpu-perf-tune/blob/25d74d031732e2f76d30a43e44c3434b588f06b3/plugins/profile-and-optimize/skills/inference-workload-profile/SKILL.md)，2026-10-31 核对。

# inference-workload-profile

## Purpose

Make speculative-decoding draft training **adaptive** by characterizing your
*own* request traffic, instead of training on a generic UltraChat+ShareGPT
mix. A draft head's acceptance (and therefore the latency win) is a function of how
well its training distribution matches the served distribution. This skill produces
the profile that the corpus builder + `inference-spec-decode-train` consume so the
draft is matched to the workload. It is an analog of Fireworks
FireOptimizer's "profile-driven customization".

## When to use

- Before training a draft head (EAGLE3/DFlash) for a model whose serving traffic has a
  distinctive shape (heavy code, heavy rewrites, long context, narrow domain).
- To decide *which* spec-dec method fits a workload (Predicted Outputs for
  rewrite-heavy traffic, EAGLE3 for general, MTP if a built-in head exists).
- As the first phase of an adaptive spec-decode loop
  (profile -> corpus -> train -> A/B).

Do **not** use this for: a model with no draft-head plan (use the standing config),
choosing serving hardware/quant (that is `inference-model-optimize` Phases 1-5).

## Workflow

The two tools ship self-contained in this skill's [`tools/`](https://github.com/cfregly/gpu-perf-tune/blob/25d74d031732e2f76d30a43e44c3434b588f06b3/plugins/profile-and-optimize/skills/inference-workload-profile/tools) dir
(`workload-profile.py`, `profile-to-corpus.py`).

### Phase 1: collect a representative profile source

Either (a) an OpenAI-style access JSONL (one request/response per line. Accepted shapes
`{"messages":[...], "completion":"..."}` or pre-counted
`{"prompt_tokens":N,"completion_tokens":M,"content_class":"..."}`. Aim for a sample
spanning a full traffic cycle, >= a few thousand requests), or (b) the
**Artificial Analysis shapes** (1k/10k/100k) when matching a draft to AA-style traffic
with no access log -- use `--aa-shapes`.

### Phase 2: run the profiler

```bash
# (a) from an access log:
python3 tools/workload-profile.py --in access.jsonl --out workload-profile.json \
  [--tokenizer <served-hf-id>]   # exact token counts; else ~4 chars/token
# (b) from the AA shapes (no access log):
python3 tools/workload-profile.py --aa-shapes --out workload-profile.json
```

Output `workload-profile.json`: input/output token distributions (mean, p50/p90/p99),
`content_class_mix`, `isl_buckets`/`osl_buckets`, `bench_shapes` (for the downstream
perf-bench), and `recommended_spec_decode.method` + rationale.

### Phase 3: hand off to the corpus builder + trainer

```bash
python3 tools/profile-to-corpus.py --profile workload-profile.json \
  [--traffic redacted.jsonl] --out corpus.jsonl     # mode b: direct redacted traffic
# OR (mode a) emit a weighted SpecForge prepare_data plan matched to the class mix.
```

Then the corpus is the `DATA_PATH` for `inference-spec-decode-train`
(SpecForge `run-offline.sh`). If the recommendation is `predicted_outputs`, there is
nothing to train -- route to the Predicted Outputs path instead.

## Privacy / discipline

- **Aggregates only by default.** The profiler stores distributions + class counts, not
  raw prompt/response text. `--keep-samples N` retains N *redacted* exemplars per class
  for the corpus builder. Use only on data the operator confirms is safe to retain.
- **Traffic is sensitive.** Redact PII / secrets upstream before any sample leaves the
  serving boundary. Record provenance (source, window, redaction) in the evidence
  bundle's SOURCE.md.
- **No external posting.** Profiles and corpora stay within the workspace. Sharing
  outside is an explicit per-turn operator decision.
- **Measured-only downstream.** The profile recommends a method. It does NOT assert a
  speedup. The acceptance + TPOT win is proven later by the
  `inference-spec-decode-train` Phase-5 A/B (same-node, >=3 trials, in-engine
  acceptance, cudagraph), per the DRAFT-vs-VERDICT rule in `docs/METHODOLOGY.md`.

## Source-of-truth references

- [`inference-spec-decode-train`](../inference-spec-decode-train/GUIDE.md) -- the train
  loop the matched corpus feeds.
- [`inference-model-optimize`](../inference-model-optimize/GUIDE.md) -- the orchestrator
  whose Phase 6 (spec-decode) this profiling step front-runs.

## Full-context reporting (no bare numbers)

Per the "Full-context reporting" rule in `docs/METHODOLOGY.md` ("every performance
number carries its full context - no bare numbers"): every number this
skill emits MUST carry its full measurement-context descriptor, and every comparison MUST be
matched on it. A bare `tok/s` / TPOT / BW / %SoL / speedup is a defect - it cannot set a
default, ship a config, or appear in a report.
- **Identity:** model (+HF path), hardware (exact ceiling token `GB300`/`B200`), quant, kv-cache dtype.
- **Parallelism:** TP, DP (replicas), PP, EP, parallel_strategy.
- **Serving cfg:** max-num-seqs, max-num-batched-tokens, gpu-memory-utilization, max-model-len, cudagraph_mode/enforce_eager, async_scheduling, prefix-caching.
- **Workload:** dataset, ISL/OSL (or mean in/out tokens), concurrency, num-prompts.
- **Regime:** warm vs cold. Latency vs throughput tier.
- **Stack:** image/vllm commit, bench backend, serving engine.
- **Grounding:** `%SoL` (+ ceiling key from `configs/sol-ceilings.yaml` - never inline a peak), sol_rigor (L1-L4), trials n (mean±std), same-node, baseline named.
- **Per-number exact shape (no smoothing):** when reporting more than one number, keep EACH with its own exact shape (ISL/OSL, concurrency, dataset, regime) - never normalize a set to one uniform descriptor that hides per-point variation (e.g. `c=1 @ ISL1024/OSL256` + `c=64 @ ISL4096/OSL512`, NOT one shared "random").

If this skill produces a measurement (tok/s, latency, %SoL, speedup), follow the
rigor discipline in `docs/METHODOLOGY.md`: capture L1 zymtrace + L3 DCGM (L4 ncu
where feasible) Speed-of-Light and publish `--strict`. Skills that
do not produce measurements are exempt.

## Next lever / BREAKTHROUGH (Grind Mandate)

If this skill emits a measured result, its output MUST end by naming the **next perf lever**,
its **expected unlock** (direction + rough magnitude), and the **gate** that proves/refutes it,
per the Grind Mandate in `docs/METHODOLOGY.md`. A
measured win is the new floor, not the finish -- so **do everything we can to find the next
BREAKTHROUGH**: the highest-EV unlock toward Speed-of-Light (a new champion / kernel / router /
quant / parallelism / spec-decode win, or an unblocked stack), not just the next micro-lever.
Rank the candidate breakthrough levers by value x cost (the GRIND FRONTIER, `perftunereport
value_view`), pursue the top, bank the rest with evidence. Record WHY a refuted lever loses,
update the standing frontier in the active bundle's `HANDOFF.md`. Never conclude
"exhausted/optimal/done" without an explicit next-lever frontier (an empty frontier AND a
documented SoL wall only). Delete this section ONLY if the skill produces no measurements.
