---
name: pytorch-refactor
description: Debug, change, or review LMDeploy's PyTorch engine to integrate a checkpoint architecture, trace per-request adapters or sampling, and diagnose state, multi-rank, or kernel behavior.
---

# LMDeploy PyTorch Engine Refactor Guide

Use these notes to locate the implementation and its callers in the user's checkout. The recorded architecture audit used `upstream/main` at `5868a57f` (2026-07-01). Check version-sensitive invariants against the target revision.

## Start Here

Start with the affected caller and use the table to locate its implementation. Follow the path across references as far as the behavior requires. `git grep <pattern> <ref> -- <paths>` and `git show <ref>:<path>` inspect another revision without switching the user's checkout.

## Route Table

| User task or observed behavior | Read first | Main files to inspect |
|-----------|------------|-----------------------|
| Trace a request from admission to forward or find where streaming stops | [request lifecycle](references/request-to-forward.md), optionally [inference pipeline](references/inference-pipeline.md) | `engine/engine.py`, `engine/engine_loop.py`, `engine/inputs_maker.py`, `engine/model_agent/agent.py`, `strategies/*/step_inputs.py` |
| Token IDs, constrained output, or logprobs differ for the same request | [input and sampling contracts](references/request-boundary-sampling.md), optionally [request lifecycle](references/request-to-forward.md) Phase 6 | `engine/input_process.py`, `engine/guided_process.py`, `engine/logits_process.py`, `strategies/*/sampling.py`, `spec_decode/spec_agent.py` |
| A disconnected client keeps running, cancel hangs, or health changes after sleep/wakeup | [runtime and distributed execution](references/runtime-adapters-distributed.md) runtime section | `engine/engine_instance.py`, `engine/mp_engine/*`, `engine/base.py`, `engine/request.py`, `serve/core/health.py`, `serve/utils/request_cleanup.py` |
| Output changes only on a prefix hit or after a different request used the slot | [prefix caching](references/prefix-caching.md). For a hybrid model, follow the saved and restored state in [cache engine](references/cache-engine.md). Compare a fresh request, slot reuse without a hit, and a hit on the same token IDs | `paging/block_trie.py`, `paging/scheduler.py`, `paging/state_manager.py`, `engine/inputs_maker.py`, `engine/engine_loop.py`, `engine/cache_engine.py`, `messages.py`, `multimodal/data_type.py` |
| A short prompt works but the same token sequence diverges when prefill is split | [long-context chunking](references/long-context-chunking.md), optionally [request lifecycle](references/request-to-forward.md) for lifecycle | `engine/inputs_maker.py`, `engine/model_agent/agent.py`, `model_inputs.py`, `spec_decode/spec_agent.py`, `backends/*/op_backend.py` |
| Capacity is lower than expected, KV/state allocation fails, or values change after wakeup | [cache engine](references/cache-engine.md) | `engine/cache_engine.py`, `engine/executor/base.py`, `engine/engine.py`, `engine/engine_checker.py`, `engine/inputs_maker.py`, `paging/state_manager.py` |
| Prefill finishes on one worker but decode never resumes on its peer after PD handoff | [PD migration](references/disagg-pd.md), then [cache engine](references/cache-engine.md) for block layout | `disagg/*`, `engine/engine.py`, `engine/engine_loop.py`, `engine/cache_engine.py`, `engine/executor/*` |
| Sequence history, multimodal inputs, logits/routed expert history | [input and sampling contracts](references/request-boundary-sampling.md) sequence section, then [symbol index](references/symbol-index.md) lookup sections | `messages.py`, `multimodal/data_type.py`, `model_inputs.py`, VLM model files |
| VLM text alone works but mixed images/audio have wrong offsets, order, or token counts | [input and sampling contracts](references/request-boundary-sampling.md) multimodal section, then [task templates](references/task-templates.md) Template 1b | `serve/processors/multimodal.py`, `vl/media/*`, `vl/model/{base,preprocess_utils,builder}.py`, `vl/model/qwen3_omni.py`, `pytorch/models/*omni*`, `pytorch/multimodal/data_type.py` |
| `ModelInputs` or `StepInputs` fields | [task templates](references/task-templates.md) Template 2b, then [symbol index](references/symbol-index.md) lookup sections | `model_inputs.py`, `strategies/ar/model_inputs.py`, `strategies/{ar,dllm,ar_spec}/step_inputs.py`, `strategies/base/model_inputs.py` |
| Add speculative decoding (MTP/EAGLE), or fix a wrong next token after partial acceptance | [request lifecycle](references/request-to-forward.md) Phases 4.5 and 4.6, [cache engine](references/cache-engine.md) for draft cache sizing, and [symbol index](references/symbol-index.md) spec sections | `spec_decode/{__init__,base,spec_agent}.py`, `spec_decode/proposers/*`, `strategies/ar_spec/*`, `engine/model_agent/{agent,dp_utils,inputs_maker}.py`, `engine/executor/base.py`, `model_inputs.py`, `models/*_mtp.py`, `models/llama_eagle*.py` |
| Add CUDAGraph support, or eager works but replay returns stale output when inputs change | [CUDAGraph rules](references/cudagraph-rules.md) | `backends/cuda/graph_runner.py`, `models/utils/cudagraph.py`, `strategies/ar_spec/cudagraph.py`, target model file |
| Bring up an unsupported HF architecture in the PyTorch pipeline or change its module mapping | [task templates](references/task-templates.md) Template 1, then [symbol index](references/symbol-index.md) model sections | `models/module_map.py`, `models/patch.py`, `models/*.py`, `configurations/*.py` |
| Load checkpoint or LoRA weights, or trace a request that selects the wrong adapter | [runtime and distributed execution](references/runtime-adapters-distributed.md) adapter/loader section | `adapter/adapter.py`, `engine/inputs_maker.py` (`local_adapter_ids`), `models/patch.py`, `weight_loader/model_weight_loader.py`, `nn/linear/lora.py`, `backends/*/lora.py` |
| A quantized projection still calls the old kernel, or only one device/shape falls back | [linear dispatch](references/linear-dispatch.md) | `nn/linear/`, `kernels/`, backend linear impls |
| One rank works but several hang, use the wrong device, or select a different executor | [runtime and distributed execution](references/runtime-adapters-distributed.md) distributed section, then [request lifecycle](references/request-to-forward.md) Phases 1-2 | `engine/executor/*`, `distributed.py`, `devices/*`, `backends/selector.py`, `envs.py`, `check_env/*`, `config.py` |
| MoE/EP/DeepEP/EPLB or GatedDelta/NSA operator paths | [runtime and distributed execution](references/runtime-adapters-distributed.md) operator anchors, then [symbol index](references/symbol-index.md) lookup sections | `nn/moe/*`, `nn/eplb.py`, `backends/cuda/moe/*`, `backends/cuda/token_dispatcher.py`, `nn/gated_delta.py`, `backends/cuda/gated_delta_rule.py`, `nn/nsa.py` |
| PR, branch, commit, local diff, or re-review | [Code review](references/code-review.md), then the topic reference for the affected behavior | Requested diff, actual callers and consumers, relevant tests |
| Unfamiliar symbol | [symbol index](references/symbol-index.md) for lookup only | Definition and call sites in the target checkout |

Use this skill to find the LMDeploy implementation and its callers. For a backend operator, identify the selected wrapper and kernel before using the `inference-engine` performance comparison. The same skill's numerical drift and state guides apply when outputs change after batching or cache reuse. Bring the actual LMDeploy token IDs, cached boundary, logits, and state slots into those comparisons. If the only evidence for a stalled deployment is worker registration, `lmdeploy-pipeline` covers the proxy request and connection checks that can locate the failing stage.

## Minimal Architecture Map

Use the [architecture map](references/architecture-contracts.md#minimal-architecture-map) to locate the component that owns the affected state.

## Hard Invariants

Check the [architecture contracts](references/architecture-contracts.md#hard-invariants) for the component you are changing against the target checkout.

## Work Pattern

- For a fix, reuse the existing owner of the state or operation. Preserve behavior relied on by callers, including the default path when adding a backend or an optimization.
- Verify the change through the entry point that triggered it. For example, a scheduler fix that affects streaming should produce the expected terminal response at the client. Use an existing regression when it exercises the changed path. Add a test when requested, required by the repository, or needed to catch a specific uncovered regression.
- For cross-module changes, trace the affected contract from its owner (`config.py`, scheduler, `ModelInputs`, `StepInputs`, or backend) to its consumers.
- For reviews, use the code review route above. Cover the whole requested diff
  on the first pass. On later passes, follow the fixes and their effects.
- Avoid mixing scheduler policy, kernel behavior, and model mapping in one patch
  unless the feature requires a new end-to-end contract.
- Temporary task skills may track active PR progress. After a PR lands, promote
  only stable architecture contracts, route changes, invariants, and validation
  targets here; do not copy progress logs, scratch benchmarks, branch todos, or
  PR-specific implementation history.

## Verification Picks

- Import smoke: `python -c "from lmdeploy import pipeline; print('import ok')"`
- Type/syntax spot check: `python -m mypy lmdeploy/pytorch/<changed_module> --ignore-missing-imports`
- Broad PyTorch suite: `pytest tests/pytorch/ -x -v`
- For a quantized projection, [verify the selected linear path](references/linear-dispatch.md#verify-the-selected-path). Choose the scope or backend test that exercises the observed dispatch.
- Prefix/cache: `tests/pytorch/paging/test_block_trie/`,
  `tests/pytorch/paging/test_scheduler.py`
- Model agent or `ModelInputs`: exercise the affected phase. For state carried across phases, check its value before and after prefill-to-decode, merge, or reindex as applicable.
- DP/MTP spec decode: `tests/pytorch/engine/test_model_agent.py::TestDPForwardMeta`,
  `tests/pytorch/spec_decode/test_spec_agent.py`, and
  `tests/pytorch/engine/test_executor_base.py` spec-cache sizing cases.
- VLM/Omni/audio: `tests/test_lmdeploy/test_vl/test_qwen3_omni_processor.py`,
  `tests/test_lmdeploy/test_vl/test_preprocess_utils.py`, and
  `tests/test_lmdeploy/test_content_merge.py`.
