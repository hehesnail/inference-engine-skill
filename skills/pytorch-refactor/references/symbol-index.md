# LMDeploy PyTorch Symbol Index

Focused symbol index for `lmdeploy/pytorch` to speed up traceability during refactoring.
This is a lookup file: search headings or symbol names and read only the matching
section instead of loading it end to end.

## Config and Invariants

- Files: `lmdeploy/messages.py`, `lmdeploy/pytorch/config.py`,
  `lmdeploy/pytorch/engine/config_builder.py`
- Symbols:
  - `PytorchEngineConfig`
  - `BackendConfig`
  - `SchedulerConfig`
  - `CacheConfig`
  - `CacheConfig.enable_prefix_caching`
  - `CacheConfig.max_prefill_token_num`
  - `DistConfig`
  - `DistConfig.__post_init__`
  - `DistConfig.from_engine_config`
  - `ModelConfig.get_num_qkv_head_by_tp`
  - `DLLMConfig`
  - `MiscConfig.enable_chunked_prefill`
  - `SpecDecodeConfig`

Use when changing distribution, cache shape, spec/DLLM config, or engine config
translation.

Notes:
- `SpecDecodeConfig` includes `dist_config` for the draft model. `qwen3_5_mtp`
  currently clones the main dist config; other spec methods use a default draft
  config unless changed.

## Engine Entry and Request Flow

- File: `lmdeploy/pytorch/engine/engine.py`
- Symbols:
  - `Engine`
  - `Engine.__init__`
  - `Engine.from_pretrained`
  - `Engine._bind_request_manager`
  - `Engine._on_stop_session`
  - `response_reqs`
  - `_build_seq_meta`
- File: `lmdeploy/pytorch/engine/engine_loop.py`
- Symbols:
  - `EngineLoop._send_resp`
  - `EngineLoop._publish_forward_prefix_cache`
  - `EngineLoop._release_forward_prefix_cache_saves`

Use when changing initialization chain, request wiring, scheduler setup,
stop-session behavior, or finish/cancel response races.

## Executor Construction

- File: `lmdeploy/pytorch/engine/executor/__init__.py`
- Symbols:
  - `get_distributed_executor_backend`
  - `build_executor`

Use when changing backend selection (`uni/mp/ray`) or distributed constraints.

## Runtime API and MP Engine

- File: `lmdeploy/pytorch/engine/engine_instance.py`
- Symbols:
  - `EngineInstance`
  - `EngineInstance.async_stream_infer`
  - `EngineInstance._get_extra_outputs`
  - `SharedStore`
- Directory: `lmdeploy/pytorch/engine/mp_engine/`
- Symbols:
  - `build_mp_engine`
  - `MPEngine`
  - `MPEngineInstance`
  - `SessionState`
  - `EngineInstancePool`
  - `EngineWorkerBase`
  - `EngineOutputGather`
  - `RayMPEngine`
  - `ZMQMPEngine`
  - `AsyncRPCServer`
  - `AsyncRPCClient`
- Files:
  - `lmdeploy/serve/utils/request_cleanup.py`
  - `lmdeploy/serve/core/health.py`
- Symbols:
  - `cleanup_result_generators`
  - `with_request_cleanup`
  - `EngineHealthMonitor`

Use when changing public runtime streaming, cancel/end semantics, sleep/wakeup,
health/metrics, P2P runtime wrappers, Ray/ZMQ behavior, or routed expert object
reference transfer.

## Distributed, Device, Backend, Env

- File: `lmdeploy/pytorch/distributed.py`
- Symbols:
  - `DistContext`
  - `DistManager`
  - `DistGroup`
  - `get_tp_world_rank`
  - `get_dp_world_rank`
  - `get_ep_world_rank`
  - `get_dist_group`
  - `gather_by_tp_sizes`
  - `reduce_scatter_by_tp_sizes`
- Files:
  - `lmdeploy/pytorch/devices/device_manager.py`
  - `lmdeploy/pytorch/backends/selector.py`
  - `lmdeploy/pytorch/backends/base.py`
  - `lmdeploy/pytorch/envs.py`
  - `lmdeploy/pytorch/check_env/*`
- Symbols:
  - `DeviceContext`, `DeviceManager`
  - `get_backend`, `init_backend`
  - `OpType`, `OpsBackend`
  - `set_envs`, `get_all_envs`

Use when changing DP/TP/EP group semantics, device-specific backend selection,
environment-driven behavior, model/device checks, or non-CUDA backend support.

## Model Agent and Forward Path

- File: `lmdeploy/pytorch/engine/model_agent/agent.py`
- Symbols:
  - `BaseModelAgent` — orchestrates forward, manages `step_inputs` state
  - `BaseModelAgent._async_loop_background()` — consumer coroutine: model forward on `self.stream`, gets from `_in_que`
  - `BaseModelAgent._async_loop_inputs_preprocess()` — producer coroutine: H2D copies on `self.out_stream`, puts to `_in_que`
  - `_H2DTransfer` / `_H2D_TRANSFER_KEY` — H2D event plus CPU source refs for non-blocking input copies
  - `BaseModelAgent._async_step()` — main step function: prefill path (full inputs) vs decode path (delta inputs)
  - `BaseModelAgent._get_inputs_from_delta()` — applies `ModelInputsDelta` to persistent `StepInputs`
  - `model_forward` — context build with KV/state caches → model forward → output normalization
  - `cache_swapping` — swap_in/swap_out with event synchronization
  - `StepInputs` — persistent decode state across steps, wraps `ModelInputs` + `ExtraInputs` + `StoppingCriteria` + `SamplingInputsDelta`
  - `ModelInputs.global_is_decoding`
  - `StepContext.global_is_decoding`
  - `StepInputs.merge_prefill()` — absorbs prefill results into the persistent decode batch
  - `StepInputs.reindex()` — index-selects surviving sequences after scheduling
  - `StepInputs.step_decode()` — advances decode state by one step
  - `BatchedOutputs`, `BatchedLogProbs`
  - DP helpers: `engine/model_agent/dp_utils.py::DistGatherScalar`,
    `DPForwardMeta`, `GatheredDPForwardMeta`
  - Streams: `self.stream` (forward), `self.out_stream` (H2D) — two CUDA streams, one asyncio event loop, requires explicit sync

Use when refactoring forward loop, step input management, cache interactions, or sampling interface.

## ModelInputs and ModelInputsDelta

- File: `lmdeploy/pytorch/model_inputs.py`
- `ModelInputs` — central dataclass for one forward batch. Core tensor fields:
  - Batch-dim-0: `seq_length`, `history_lengths`, `block_offsets`, `num_ignored_history`, `local_adapter_ids`, `state_offsets`
  - Token-dim (-1): `input_ids` shape `(1, sum_tokens)`
  - Special-dim (1): `mrope_pos_ids` shape `(3, sum_seqlens)`
  - Spec/draft fields: `target_hidden_states`, `target_position_ids`,
    `target_inputs_embeds` (verify shape and paradigm handling before indexing)
  - Aggregates: `is_decoding`, `max_q_seqlen`, `max_kv_seqlen`, `sum_kv_seqlen`
  - List: `model_metas` (CPU-indexed)
  - Non-indexable/lifecycle flags: `vision_inputs`, `dp_meta`, `enable_microbatch`, `is_dummy`, `is_chunk`, `is_first_chunk`, `is_last_chunk`, `is_chunk_multimodal`
- `DPMeta` — TP size lists plus spec/DP coordination fields:
  `is_decoding`, `dp_is_decoding`, `dp_batches`, `dp_draft_num_tokens`.
- `ModelInputs.step()` — in-place advance for decode (updates `history_lengths`, `max_kv_seqlen`, `sum_kv_seqlen`, `input_ids`, `mrope_pos_ids`)
- `ModelInputs.to_device()` — generic via `fields(self)`
- `ModelInputsDelta` (line ~125) — NOT a diff of tensor values but an **index-select specification**: `indices`, `block_offsets`, `max_kv_seqlen`, etc.

Use when adding/modifying fields in ModelInputs or changing merge/index_select/step behavior.

## Sequence History and Multimodal Data

- File: `lmdeploy/pytorch/messages.py`
- Symbols:
  - `InputEmbeddings`
  - `SamplingParam`
  - `MessageStatus`
  - `SequenceMeta`
  - `SequenceManager`
  - `SchedulerSession`
  - `HistoryEmbeddings`
  - `HistoryTokenIds`
  - `HistoryRouterExperts`
  - `HistoryLogits`
  - `HistoryMropePosIds`
  - `HistoryMultiModals`
  - `PrefixCacheMeta`
  - `PrefixCacheState`
  - `SchedulerSequence`
  - `SchedulerSequence.get_chunk_limit_multimodals`
  - `SchedulerSequence.get_prefix_cache_extra_hashes`
  - `SchedulerSequence.clamp_prefix_cache_match_step`
- File: `lmdeploy/pytorch/multimodal/data_type.py`
- Symbols:
  - `MultiModalData`
  - `MultiModalInputs`
  - `make_multimodal_content_hash`
  - `ensure_multimodal_content_hashes`

Use when changing request/session history, multimodal span tracking, MROPE
position history, returned logits, routed experts, or prefix/chunk interactions
with sequence state.

## VLM / Multimodal Preprocessing

- File: `lmdeploy/serve/processors/multimodal.py`
  - `MultimodalProcessor.async_parse_multimodal_item()`
  - `_has_multimodal_input()`
- Media files:
  - `vl/media/image.py`, `video.py`, `audio.py`, `time_series.py`
- File: `lmdeploy/vl/model/base.py`
  - `VisionModel`
  - `VisionModel.ATTR_NAME_TO_MODALITY`
  - `VisionModel.FEATURE_NAMES`
  - `VisionModel.preprocess()`
  - `MultimodalSpecialTokens`
- File: `lmdeploy/vl/model/preprocess_utils.py`
  - `get_expanded_input_ids`
  - `get_expanded_mm_items`
  - `get_mm_items_offset`
  - `_expand_bundled_image_items`
  - `_expand_bundled_video_items`
  - `_expand_bundled_audio_items`
- Qwen3 Omni:
  - `vl/model/qwen3_omni.py::Qwen3OmniModel`
  - `pytorch/configurations/qwen3_omni.py::Qwen3OmniModelConfigBuilder`
  - `pytorch/models/qwen3_omni_moe_thinker.py::Qwen3OmniMoeThinkerForConditionalGeneration`
  - `pytorch/models/qwen3_omni_moe_thinker.py::Qwen3OmniInputProcessor`

Use when changing OpenAI multimodal request parsing, audio/video/image
processor outputs, multimodal token expansion, MROPE ids, or Qwen3 Omni
thinker support.

## Input Makers

- File: `lmdeploy/pytorch/engine/inputs_maker.py`
- Symbols:
  - `InputsMakerConfig`
  - `LongContextChunker`
  - `InputsMakerAsync.create_model_inputs()`
  - `InputsMakerAsync.create_model_inputs_long_context()`
  - `InputsMakerAsync.create_model_inputs_delta()`
  - `InputsMakerAsync._make_forward_inputs()`
  - `InputsMakerAsync.has_pending_long_context_chunk()`
  - `InputsMakerAsync.do_prefill_default()`
  - `InputsMakerAsync.do_prefill_chunked()`
- File: `lmdeploy/pytorch/engine/model_agent/inputs_maker.py`
- Symbols:
  - `DefaultForwardInputsMaker`
  - `DPForwardInputsMaker`
  - `build_inputs_maker`

Use when changing scheduler-facing input creation, worker-side DP
synchronization, queueing, or dummy-filled inputs.

## Scheduler, Prefix Cache, and Long-Context Chunking

- File: `lmdeploy/pytorch/paging/scheduler.py`
- Symbols:
  - `Scheduler._schedule_prefill()` — prefix-cache match, admission cap, allocation
  - `Scheduler._rollback_unscheduled_prefix_match()`
  - `Scheduler._check_prefill_admission_gates()` — short/long prefill gate checks, including tentative prefix-match shrinkage
  - `Scheduler.reserve_long_context_chunk()` — allocates/reserves KV for active long-context chunk continuations
  - `Scheduler._finish_prefix_cache_schedule()`
  - `Scheduler._schedule_decoding()` — decode allocation and trie registration
  - `Scheduler._schedule_migration()` — PD migration admission/allocation
  - `Scheduler.schedule_running()` — allocates additional blocks for persistent decode batches
  - `Scheduler.schedule_metrics` — exposes `prefix_cache_hit_rate`
- File: `lmdeploy/pytorch/paging/block_trie.py`
- Symbols:
  - `BlockTrie.match()` — finds shared full-block prefixes and calls `seq.set_step()`
  - `BlockTrie.allocate()` — registers newly computed full blocks and handles duplicate blocks
  - `BlockTrie.evict()` — evicts trie-owned leaf blocks with ref count 1
  - `BlockTrie.reserve_state_checkpoint_for_seq()`
  - `BlockTrie.reserve_decode_state_checkpoint_for_seq()`
  - `BlockTrie.commit_state_checkpoint_for_seq()`
  - `BlockTrie.acquire_state_checkpoint_restore_for_seq()`
  - `BlockTrie.release_state_checkpoint_restore_for_seq()`
  - `BlockTrie.release_state_checkpoint_save_for_seq()`
  - `BlockTrie.evict_state_checkpoints()`
  - `PrefixCacheStats`
- File: `lmdeploy/pytorch/paging/state_manager.py`
- Symbols:
  - `StateManager`
  - `StateManager.allocate()`
  - `StateManager.reserve_checkpoint()`
  - `StateManager.get_num_free_runtime()`
  - `StateManager.get_num_free_checkpoint()`
- File: `lmdeploy/pytorch/engine/cache_engine.py`
- Symbols:
  - `StateCacheEngine.copy_caches`
- File: `lmdeploy/pytorch/engine/inputs_maker.py`
- Symbols:
  - `LongContextChunker`
  - `InputsMakerAsync.create_model_inputs_long_context()`
  - `InputsMakerAsync._make_forward_inputs()`
  - `InputsMakerAsync.do_prefill_default()`
  - `InputsMakerAsync.do_prefill_chunked()`

Use when changing text/VLM/SSM prefix-cache reuse, prefill admission, current
long-context chunking, or the unwired `enable_chunked_prefill` path. See
`references/prefix-caching.md` or `references/long-context-chunking.md`.

## Strategy Dispatch

- File: `lmdeploy/pytorch/strategies/__init__.py`
- Symbols:
  - `build_strategy_factory` — dispatches on `model_paradigm`: `ar` / `dllm` / `ar_spec`

### Strategy ABCs (all in `strategies/base/`)

| ABC | File | Key Methods |
|-----|------|-------------|
| `ModelInputsStrategy` | `base/model_inputs.py` | `create_make_dummy_meta()`, `make_dummy()`, `make_dummy_extra_inputs()` |
| `ModelAgentStrategy` | `base/model_agent.py` | `slice_outputs()`, `slice_extra_inputs()`, `make_stopping_criteria()`, `make_extra_inputs()`, `make_extra_outputs()`, `step_sampling_inputs()`, `post_sampling()`, `broadcast_next_token()`, `post_broadcast()` |
| `EngineStrategy` | `base/engine.py` | `get_num_decode_tokens()`, `get_num_loops()`, `get_prealloc_size()` |
| `SequenceStrategy` | `base/sequence.py` | `make_sequence()`, `update_running()` |
| `SamplingStrategy` | `base/sampling.py` | `make_sampling_inputs()`, `on_session_end()` |
| `CudagraphStrategy` | `base/cudagraph.py` | `get_max_tokens()` |
| `StepInputs` | `base/step_inputs.py` | `merge_prefill()`, `reindex()`, `step_decode()` |

### Strategy Concrete Implementations

| Paradigm | Directory | `ExtraInputs` | `StoppingCriteria` | `tokens_per_step` |
|----------|-----------|----------------|---------------------|--------------------|
| AR | `strategies/ar/` | empty | `num_appendable_ids` | 1 |
| DLLM | `strategies/dllm/` | `dllm_mask` tensor | `num_appendable_ids` + `output_start_pos` | `block_size` |
| AR_Spec | `strategies/ar_spec/` | `target_logits`, `output_draft_token_ids`, etc. | extends AR + multi-token | `num_spec_tokens + 1` |

### Key Strategy Functions

- `get_model_inputs_next_decoding()` in `ar/model_inputs.py` and `_get_model_inputs_next_decoding()` in `dllm/step_inputs.py` — creates first decode-step `ModelInputs` from prefill output
- `merge_model_inputs()` in `ar/model_inputs.py` — concatenates two `ModelInputs` batches (shared by all paradigms)
- `index_select_model_inputs()` in `ar/model_inputs.py` — AR decode reindex helper
- `_reindex_model_inputs_dllm()` in `dllm/step_inputs.py` — DLLM decode reindex helper
- `_reindex_model_inputs_arspec()` in `ar_spec/step_inputs.py` — AR_Spec decode reindex helper

### Known Watch Points
- Reindex and prefill-to-decode conversion are duplicated across AR, DLLM, and AR_Spec. New `ModelInputs` fields are easy to miss.
- DLLM prefill-to-decode and reindex helpers do not currently carry `mrope_pos_ids`.
- AR_Spec reindex handles `mrope_pos_ids` but does not carry
  `target_hidden_states`, `target_position_ids`, or `target_inputs_embeds`;
  verify assumptions before adding fields used by spec decode.

Use when adding/changing model paradigm branches (`ar`, `dllm`, `ar_spec`) or modifying step input flows.

## Model Rewrite Mapping

- File: `lmdeploy/pytorch/models/module_map.py`
- Symbols:
  - `MODULE_MAP`
  - `DEVICE_SPECIAL_MODULE_MAP`
  - `ASCEND_MODULE_MAP`
  - `MACA_MODULE_MAP`
  - `CAMB_MODULE_MAP`
  - `CUSTOM_MODULE_MAP`

Use when adding architecture mappings or device-specific rewrites.

## Model Patching and Adapter Injection

- File: `lmdeploy/pytorch/models/patch.py`
- Symbols:
  - `_get_module_map`
  - `update_custom_module_map`
  - `_get_model_class`
  - `build_model_from_hf_config`
  - `build_patched_model`
  - `add_adapters`

Use when changing class resolution, build context, or LoRA adapter integration.

## Adapters and Weight Loading

- File: `lmdeploy/pytorch/adapter/adapter.py`
- Symbols:
  - `AdapterManager`
  - `find_all_target`
  - `get_ranks_and_scalings`
  - `load_lora_weights`
- File: `lmdeploy/pytorch/weight_loader/model_weight_loader.py`
- Symbols:
  - `load_weight`
  - `default_weight_loader`
  - `ModelWeightLoader`
  - `load_model_weights`
- Related files:
  - `lmdeploy/pytorch/nn/linear/lora.py`
  - `lmdeploy/pytorch/backends/*/lora.py`
  - `lmdeploy/pytorch/kernels/cuda/fused_lora.py`

Use when changing LoRA adapter id mapping, packed projection adapter loading,
custom `param.weight_loader` behavior, checkpoint loading, model
`rename_weight()`, or post-load `update_weights()` hooks.

## Spec Decode Entry

- File: `lmdeploy/pytorch/spec_decode/__init__.py`
- Symbols:
  - `build_spec_agent`
- File: `lmdeploy/pytorch/spec_decode/base.py`
- Symbols:
  - `BaseSpecModelAgent`
  - `_build_draft_dist_ctx`
  - `get_padding_batch_size`
- File: `lmdeploy/pytorch/spec_decode/spec_agent.py`
- Symbols:
  - `SpecModelAgent`
  - `_prepare_inputs_from_main`
  - `_build_dp_meta_from_main`
  - `_rejection_sampling`
  - `_async_model_forward`
- Directory: `lmdeploy/pytorch/spec_decode/proposers/`

Use when touching draft-model path or speculative decoding enable conditions.

## DisAgg / PD Migration

- Directory: `lmdeploy/pytorch/disagg/`
- Symbols:
  - `EngineRole`, `ServingStrategy`, `MigrationBackend`
  - `EngineP2PConnection`
  - `PDConnectionPool`
  - `MigrationExecutionBatch`, `MigrationAssignment`
- Engine/cache symbols:
  - `Engine.p2p_initialize()`, `p2p_connect()`, `p2p_drop_connect()`
  - `EngineLoop.migration_loop()`
  - `CacheEngine.p2p_initialize()`, `p2p_connect()`, `migrate()`

Use when touching Prefill/Decode roles, proxy connection setup, P2P KV transfer,
or migration scheduler behavior. See `references/disagg-pd.md`.

## Request Preprocessing, Sampling, Guided Decoding

- File: `lmdeploy/pytorch/engine/input_process.py`
- Symbols:
  - `PreprocessInputResult`
  - `BaseModelInputProcessor`
  - `DefaultModelInputProcessor`
- File: `lmdeploy/pytorch/engine/guided_process.py`
- Symbols:
  - `GuidedDecodingManager`
- File: `lmdeploy/pytorch/engine/logits_process.py`
- Symbols:
  - `FusedLogitsProcessor`
  - `SamplingInputs`
  - `SamplingInputsDelta`
- Files:
  - `lmdeploy/pytorch/strategies/{ar,dllm,ar_spec}/sampling.py`
  - `lmdeploy/pytorch/spec_decode/spec_agent.py`
- Symbols:
  - `make_sampling_inputs`
  - `_expand_sampling_inputs`
  - `_slice_sampling_inputs`
- Serve/OpenAI extensions:
  - `ChatCompletionRequest.return_logprob`
  - `ChatCompletionResponseChoice.output_token_logprobs`
  - `ChatCompletionResponseStreamChoice.output_token_logprobs`
  - `api_server.py::_create_output_token_logprobs`

Use when changing request preprocessing, response formats/guided decoding,
custom logits processors, repetition penalty/ngram, top-k/top-p/min-p,
logprobs, raw output token logprobs, output logits, or sampling state carried
through `StepInputs`.

## MoE, EP, DeepEP, EPLB

- Files:
  - `lmdeploy/pytorch/nn/moe/*`
  - `lmdeploy/pytorch/nn/eplb.py`
  - `lmdeploy/pytorch/backends/cuda/moe/*`
  - `lmdeploy/pytorch/backends/cuda/token_dispatcher.py`
  - `lmdeploy/pytorch/backends/deepep_state.py`
- Symbols:
  - `MoeType`
  - `DispatchInputs`
  - `FusedMoEBase`
  - `MoEForwardDPTP`
  - `EPLBManager`
  - `DeepEPTokenDispatcher`
  - `DeepEPTokenDispatcherLowLatency`

Use when changing expert parallel routing, MoE TP/DP_TP behavior, DeepEP
prefill/decode dispatch, EPLB logical/physical expert ids, or MoE fused kernel
builders.

## GatedDelta, SSM State, NSA

- Files:
  - `lmdeploy/pytorch/nn/gated_delta.py`
  - `lmdeploy/pytorch/backends/cuda/gated_delta_rule.py`
  - `lmdeploy/pytorch/backends/gated_delta_rule.py`
  - `lmdeploy/pytorch/kernels/cuda/gated_delta_preprocess.py`
  - `lmdeploy/pytorch/kernels/cuda/gated_delta_rule.py`
  - `lmdeploy/pytorch/nn/nsa.py`
  - `lmdeploy/pytorch/backends/cuda/nsa.py`
- Symbols:
  - `GatedDeltaMeta`
  - `GatedDelta`
  - `CausalConv1d`
  - `load_state`
  - `GatedDeltaRuleImpl.prepare_inputs`
  - `gated_delta_preprocess`
  - `IndexerTopKFP8`
  - `TritonNSAIndexFP8`

Use when changing SSM/state-cache model behavior, spec offsets for recurrent
state, gated delta kernels, causal conv state loading, or NSA FP8 indexing.

State allocation lookup:
- File: `lmdeploy/pytorch/paging/state_manager.py`
- Symbols: `StateAllocator`, `StateManager`, `build_state_manager`
- State id 0 is reserved; `num_state_caches` includes the reserved row.

## Kernel and NN Layers (Performance-Sensitive)

- Directories:
  - `lmdeploy/pytorch/nn/`
  - `lmdeploy/pytorch/kernels/`

Use only when the issue is proven to be at operator/kernel level; avoid mixing kernel and scheduler refactors in one patch.
