> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[cfregly/gpu-perf-tune@25d74d031732](https://github.com/cfregly/gpu-perf-tune/blob/25d74d031732e2f76d30a43e44c3434b588f06b3/plugins/profile-and-optimize/skills/inference-known-good-config/SKILL.md)，2026-10-31 核对。

# inference-known-good-config

## Purpose

Make per-model serving-config knowledge **durable, queryable, and enforced**. The
serve flags a model REQUIRES to run correctly (not its full champion config -- the
boot-blockers / crash-avoiders / deploy-correctness workarounds) are otherwise scattered
across bring-up docs, `my-values-*.yaml`, `arms.tsv`, and cross-engine sweep docs, so
the next operator re-discovers them the hard way. This skill captures them in ONE
registry and makes a deploy that DROPS a required flag a fail-closed error.

The motivating failure: **Qwen3-Next on vLLM 0.22** auto-selects the FlashInfer
Blackwell GDN *prefill* kernel, which WEDGES at c>=~8 on the interleaved-GQA layout
(decode->0, EngineCore hang). The fix is one flag --
`--additional-config '{"gdn_prefill_backend":"triton"}'` -- that is **not a
well-known combination**. It is a field-discovered workaround. Without a
registry it would be re-discovered each time the model is redeployed.

Backed by two native MCP verbs:
- `known_good_config_record` -- append a NEW model entry (comment-preserving).
- `known_good_config_check` -- diff a deploy's serve args vs the registry, **fail-closed**
  (nonzero) on a missing boot-blocker / crash-high-c / deploy-correctness flag.

## When to use

- **After finding a champion** (end of `inference-tune-sweep` / `inference-model-optimize`):
  `record` the model's required flags + champion pointer + evidence. This is part of the
  grind-closure gate -- a champion is not "closed" until its known-good config is registered
  (see "Next lever (the grind ratchet)" below).
- **Before any deploy/ship**: `check` the deploy's serve args (or its `-f` manifest) vs the
  registry. A missing required flag is fail-closed. To run the check automatically on a
  `vllm serve` / `kubectl apply`, wire `known_good_config_check` as a pre-exec guard hook
  using the runtime-agnostic hook contract in
  [`hooks/README.md`](https://github.com/cfregly/gpu-perf-tune/blob/25d74d031732e2f76d30a43e44c3434b588f06b3/plugins/profile-and-optimize/hooks/README.md) (this repo does not ship
  that guard pre-wired. The two shipped gates there show the pattern).
- When an operator asks "what flags does <model> need" / "did we capture that workaround".

Do NOT use this for the FULL champion config -- that stays in `my-values-<slug>.yaml` /
the deploy YAML (the registry points at it via `champion.config_ref`, never duplicates it).

## Example prompts

- "Record the Qwen3-Next known-good config: it needs gdn_prefill_backend=triton on 0.22."
- "Check this deploy file against the known-good registry before I apply it."
- "What required flags does MiniMax-M2.7 need at high concurrency?"
- "Did we capture the gdn_prefill_backend=triton workaround anywhere durable?"

## Prerequisites

1. **Registry** -- `perf-tune-report/configs/known-good-configs.yaml` (schema `known_good_config_v1`).
   Resolved via `--registry`, then `$KNOWN_GOOD_CONFIG_REGISTRY`, then a walk-up from cwd.
2. **Model id** -- the HF id / served-model-name (the registry key).
3. For `check`: the deploy's serve args (`--serve-args "<joined args>"`) OR a `--deploy-file`.

## Interaction style

Autonomous for `check` (it is a gate. Run it). One pause for `record`: confirm the
required-flag tuple(s) + evidence path before the append.

## Workflow

### record (after a champion / a newly-discovered required flag)

```text
mcp__profile_and_optimize__known_good_config_record with:
  args: ["--model", "<hf-id>",
         "--slug", "<bundle-slug>", "--arch", "<arch note>", "--hardware", "<hw>",
         "--engine", "vllm",
         "--required-flag", "<flag>|<match-regex>|<severity>|<why>|<affected>|<evidence-path>",
         "--champion-config-ref", "<my-values / deploy yaml path>",
         "--champion-verdict", "DRAFT <n> | VERDICT <n>",
         "--champion-campaign", "<campaign-id>",
         "--grind-frontier", "value-findings.yaml -> <model> next_lever",
         "--json"]
```

- `severity` is one of `boot-blocker | crash-high-c | deploy-correctness | perf`. The first three
  are fail-closed in `check`, `perf` is a warning.
- `record` is **append-only + comment-preserving**. An EXISTING model fails with guidance to
  edit the YAML by hand (a programmatic rewrite would strip the LOUD banner + per-entry prose).
  Updating a champion verdict (DRAFT->VERDICT) is a hand-edit.

### check (before deploy/ship. Fail-closed)

```text
mcp__profile_and_optimize__known_good_config_check with:
  args: ["--model", "<hf-id>",
         "--deploy-file", "<deploy.yaml>",      # OR --serve-args "<joined args>"
         "--json"]
```

- Returns `verdict: pass|fail` + `missing_required: [...]`. A missing
  boot-blocker/crash-high-c/deploy-correctness flag -> `fail` + nonzero exit (the gate blocks).
- An unregistered model also returns `fail` with a nonzero exit. Record and
  review the model before treating its deployment as known-good.

## Registry layout

```
perf-tune-report/configs/known-good-configs.yaml   # schema: known_good_config_v1
  models:
    - model: <HF id>            # the key
      slug, arch, hardware, engine
      required_flags:           # boot-blocker / crash-high-c / deploy-correctness / perf
        - {flag, match, severity, why, affected, evidence}
      champion: {config_ref, verdict, campaign}   # pointer, NOT the full config
      fallback: <a known-working alternative>
      grind_frontier: <cross-ref into value-findings.yaml next_lever>
```

## Safety

- **Keep the registry private.** Never post a registry entry (flags, model names, a
  not-yet-reported upstream bug) to a public repo / upstream / chat without explicit
  per-turn operator approval.
- **Append-only + comment-preserving** for `record`. The LOUD banner + per-entry `why` prose are
  load-bearing and must survive.
- **Fail-closed `check`** is the point -- do not work around a `fail` by stripping the flag from
  the registry. Either add the flag to the deploy, or (if it is a genuinely superseded
  requirement) hand-edit the registry with evidence.

## Verdict rigor (DRAFT vs VERDICT)

Per `docs/METHODOLOGY.md` "Verdict rigor: DRAFT vs VERDICT", a
`champion.verdict` is **DRAFT** until variance-controlled (same-node, >=3 trials, mean+/-std,
metric-isolated, fair baseline). Record the honest tier in the entry. Promote DRAFT->VERDICT
by a hand-edit once the controlled A/B lands.

## Next lever (the grind ratchet)

Per `docs/METHODOLOGY.md` "Always be grinding":
every `record` MUST set `grind_frontier` (the cross-ref into
`configs/value-findings.yaml` `next_lever`). A
known-good config is the CONFIG half of closure, `value-findings.yaml` is the GRIND half. The
grind-closure gate checks BOTH before a champion is "closed":
`known_good_config_check` for the CONFIG half, and a recorded `next_lever` in
`value-findings.yaml` for the GRIND half.

## Source-of-truth references

- `configs/known-good-configs.yaml` -- the registry.
- `configs/value-findings.yaml` -- the paired GRIND FRONTIER (`next_lever`).
- `docs/METHODOLOGY.md` -- verdict rigor + the grind ratchet (the CONFIG-half rule).
- [`server/tools/known_good_config/known_good_config_cli.py`](https://github.com/cfregly/gpu-perf-tune/blob/25d74d031732e2f76d30a43e44c3434b588f06b3/plugins/profile-and-optimize/server/tools/known_good_config/known_good_config_cli.py) -- the verb implementation.
