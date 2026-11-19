# LMDeploy PyTorch architecture contracts

Guide to the LMDeploy PyTorch engine at upstream/main commit `5868a57f` (2026-07-01). Start with the contract the change touches and confirm the details in the target revision.

## Minimal Architecture Map

- Config enters through `lmdeploy/messages.py::PytorchEngineConfig`, then
  `engine/config_builder.py` builds `SchedulerConfig`, `CacheConfig`,
  `BackendConfig`, `DistConfig`, `MiscConfig`, and optional `SpecDecodeConfig`
  in `config.py`.
- `SpecDecodeConfig` carries its own draft `DistConfig`: `qwen3_5_mtp` clones
  the main dist config, while other spec methods still use a default draft
  config unless explicitly redesigned.
- `ModelConfig` retains its `DistConfig`; use
  `ModelConfig.get_num_qkv_head_by_tp()` for TP-local attention head counts in
  backend metadata, graph buffers, FlashMLA, and FA3 paths.
- Engine flow is `Engine` -> `build_executor()` -> strategy factory -> `Scheduler` ->
  `EngineLoop` -> `InputsMakerAsync` -> executor workers -> `BaseModelAgent`.
- External runtime calls use `Engine.create_instance()` -> `EngineInstance`
  for in-process serving, or `engine/mp_engine/*` for Ray/ZMQ multi-process
  wrappers.
- Core input maker is `engine/inputs_maker.py`. `engine/model_agent/inputs_maker.py`
  is the worker-side/DP/dummy forward-input helper, and
  `engine/model_agent/dp_utils.py` owns DP scalar metadata gathering.
- Request/session state lives in `messages.py`: `SamplingParam`,
  `SchedulerSequence`, token/history containers, multimodal history, logits
  history, routed expert history, and prefix-cache sequence metadata.
- VLM request preprocessing spans `serve/processors/multimodal.py`,
  `vl/media/*`, `vl/model/*`, and model-specific
  `BaseModelInputProcessor` implementations. Audio is now a first-class
  `Modality` alongside image/video/time-series.
- Persistent decode state lives in strategy `StepInputs`; full batch inputs live
  in `model_inputs.py::ModelInputs`, and scheduling reindex data is
  `ModelInputsDelta`.
- `InputsMakerAsync` owns TTFT-oriented long-prefill policy around the scheduler:
  short/normal prefill turns can be admitted before waiting long prompts,
  active long chunks are engine-local runnable work, and scheduler calls now
  distinguish `allow_long_prefill` and `prefer_long_prefill`.
- Model paradigm dispatch is `strategies/__init__.py`: `ar`, `dllm`, `ar_spec`.
- Disaggregated serving roles/backends live in `pytorch/disagg`; migration
  enters the engine through `EngineRole`, `EngineP2PConnection`, scheduler
  migration states, and `CacheEngine.p2p_*()/migrate()`.
- Backend abstraction is `backends/base.py::OpType` plus
  `OpsBackend.get_layer_impl_builder()`. Models should use local builders such
  as `build_qkv_proj()`/`build_o_proj()` instead of direct backend construction.
- Device and distributed contexts are process-global context managers in
  `devices/device_manager.py` and `distributed.py`; backend selection reads the
  current device context through `backends/selector.py`.
- Model rewrite resolution is `models/module_map.py` plus `models/patch.py`;
  architecture-specific config builders auto-register from `configurations/`.

## Hard Invariants

- Cache swap must wait on `cache_engine.events.wait()`.
- Cache sizing reserves state cache and runtime memory before applying
  `cache_max_entry_count`; explicit `num_gpu_blocks` must not be re-estimated.
- State cache ids reserve row 0 for system/padding use. `num_state_caches`
  includes that reserved row; `StateManager` allocates ids from 1 through
  `num_state_caches - 1`.
- `block_size` must be divisible by `kernel_block_size`; PD migration currently
  requires equality.
- Current PD migration is KV-block oriented. Do not assume SSM/state-cache
  models such as Qwen3.5 work with PD until state-cache handoff, migration or
  recompute policy, scheduler admission/allocation, and validation are designed.
- Prefix cache is text/VLM/SSM-aware in the PyTorch engine and can be used by
  AR-spec/MTP. It remains disabled for PD/disaggregated roles, sliding-window
  attention, TurboMind VLM, and legacy PyTorch VLM paths unless each path is
  explicitly redesigned and tested.
- AR-spec/MTP prefix-cache matches keep a recompute overlap:
  `SchedulerSequenceARSpec.__post_init__()` sets
  `prefix_cache.match_recompute_blocks=1`; `BlockTrie.match()` drops that
  overlap from the accepted hit and `BlockTrie.allocate()` keeps the overlap
  blocks private/writable instead of substituting shared trie blocks.
- VLM prefix caching is limited to PyTorch new-preprocess
  `HistoryMultiModals`. Prefix-cache block keys include overlapping
  `(start, end, modality, content_hash)` entries so identical placeholder
  tokens backed by different media do not collide. Matching must clamp to whole
  multimodal spans.
- SSM prefix caching requires an exact ready state checkpoint for the matched
  trie node. KV-only SSM matches are invalid; checkpoint restore/save uses
  compact host integer state-copy offsets and `StateCacheEngine.copy_caches()`
  on the forward stream.
- Prefix cache order is tentative `block_trie.match()` -> eviction/state
  admission -> `block_manager.allocate()`/`state_manager.allocate()` ->
  `block_trie.allocate()` -> finish/publish. If scheduling cannot accept the
  sequence, rollback the tentative match, restore stats, release pinned state
  checkpoints, and reset cached-token accounting.
- Prefix-cache cached-token accounting is per current prompt:
  `seq.cached_tokens` excludes previous-session history, recompute/preemption
  replay, rolled-back tentative matches, and AR-spec private recompute overlap.
- Current long-context chunking is not official chunked prefill: chunks are not
  computed together with decode steps; non-last chunks produce no user-visible
  output and do not merge into persistent decode `StepInputs`. The TTFT policy
  sends first-slice/eager long chunks as separate forwards, then gives decode
  and bounded short-prefill turns chances to run before continuing long work.
- Active long-context chunks are not represented only by scheduler
  WAITING/READY queues. `EngineLoop` runnable checks must include
  `InputsMakerAsync.has_pending_long_context_chunk()`.
- `StepInputs` lifecycle is `reindex()` before forward, `merge_prefill()` after
  prefill, and `step_decode()` after decode across all paradigms.
- `SamplingInputs` fields are part of the persistent decode contract when they
  flow into `SamplingInputsDelta`; update strategies and spec expand/slice
  helpers when adding tensor fields.
- Follow each new `ModelInputs` field from its producer to its last consumer. Fields that persist into decode must survive the relevant conversion, merge, and reindex paths. Carry the value through `ModelInputs.step()` or graph dummy inputs when those paths consume it.
- `ModelInputs` carries compact SSM prefix-cache state restore/save offset
  sequences: `state_prefix_cache_offsets`,
  `state_prefix_cache_dst_offsets`, `state_prefix_cache_save_src_offsets`, and
  `state_prefix_cache_save_offsets`. They are not tensors to index with CUDA
  masks; keep them as compact host integer pairs.
- New multimodal modalities must wire through public request parsing, media IO,
  `VisionModel.ATTR_NAME_TO_MODALITY`, `MultimodalSpecialTokens`,
  `get_expanded_mm_items()`, `MultiModalData`, model-specific input processors,
  and `DeployModelMixinV1.get_multimodal_mask()`.
- Text-only use of hybrid VLM checkpoints is `language_model_only` on
  `PytorchEngineConfig`/`TurbomindEngineConfig` and CLI
  `--language-model-only`; old `disable_vision_encoder` wording is stale.
- Qwen3 Omni support maps `Qwen3OmniMoeForConditionalGeneration` to the
  thinker-only PyTorch model. Its config builder uses
  `thinker_config.text_config` and enables MROPE; max-len derivation must unwrap
  `thinker_config.text_config`; talker/code2wav weights are intentionally
  skipped.
- Qwen3 Omni video items are whole-video spans, not Qwen3-VL per-frame spans.
  Preserve `video_second_per_grid` and prompt-order sorting by multimodal
  offsets when changing VLM preprocessing.
- Qwen3 Omni audio uses lazy `librosa`/`soundfile` media IO, defaults to 16 kHz,
  carries `feature_attention_mask` to audio feature lengths, and creates audio
  MROPE ids over the audio token span.
- MRoPE construction is centralized in
  `nn.rotary_embedding.build_rotary_embedding_from_config()`. Use
  `MRotaryEmbedding` for both chunked and interleaved MRoPE; do not reintroduce
  model-local Qwen3/Qwen3-VL rotary table builders unless the backend contract
  changes.
- `model_forward()` builds `StepContext` before `prepare_inputs_for_generation()`
  and model forward; SSM state cache data flows through `state_caches` and
  `state_offsets`, with no `StateCacheEngine.init_caches()` hook on current main.
- Sleep/wakeup with `kv_cache` rebuilds cache engines and runs warmup inside
  `BaseModelAgent.wakeup()` before `state.is_sleeping=False` and before DP
  `to_wakeup` is released. The top-level `Engine.wakeup()` only delegates to
  the executor, clears sleeping tags, and unblocks inputs after all tags clear.
  `empty_init` skips init-time warmup, so wakeup and update-params flows must
  restore graph/cache readiness explicitly.
- CUDA streams are explicit: `out_stream` H2D/D2H and forward `self.stream`
  require synchronization. H2D now records a CUDA event in `_H2DTransfer`,
  keeps CPU source refs alive until the event completes, and waits from
  `self.stream` with `wait_event()` before forward kernels read the data.
- CUDA speculative decoding with `ar_spec` requires FA3 unless the model uses
  FlashMLA; check `backends/cuda/graph_runner.py` before changing spec decode
  graph behavior.
- For DP + spec decode, keep local and DP-global decode state separate.
  `ModelInputs.is_decoding` is local; `ModelInputs.global_is_decoding()` and
  `StepContext.global_is_decoding()` read `dp_meta.dp_is_decoding` when DP meta
  exists. Do not overwrite local `is_decoding` during DP mixed decode.
- DP + MTP draft token counts are not always the main token counts around
  long-context chunks. Preserve `dp_meta.dp_draft_num_tokens`; DP=1 non-last
  chunks may skip remaining draft forwards, but DP>1 non-last chunks must still
  run the full draft-forward loop to keep ranks aligned.
- `qwen3_5_mtp` draft cache can be TP-sharded. Cache sizing must use the draft
  `SpecDecodeConfig.dist_config.attn_tp`: draft TP=1 charges spec cache only to
  one rank per attention-TP group, draft TP>1 charges all participating ranks.
- `qwen3_5_mtp` draft input conversion uses local `is_decoding` for draft
  inputs, while padding and graph decisions use `global_is_decoding()`.
- Spec decode postprocessing is split: target logits/rejection sampling happen
  before draft forward, then draft tokens are broadcast with
  `ModelAgentStrategy.post_broadcast()`.
- CUDAGraph spec decode keys distinguish decode query length, not
  `target_hidden_size`: key shape is `(capture_batch, global_is_decoding,
  enable_microbatch, query_len)`. `CudaGraphMeta.decode_query_len` drives
  FlashMLA/FA3 metadata, and DP padding must sync TP sizes as padded batch
  times query length.
- `PytorchEngineConfig.cudagraph_capture_batch_sizes` can restrict CUDA graph
  capture sizes. Normalize to sorted unique positive ints no larger than
  `max_batch_size`, and always include `max_batch_size`; graph runners
  defensively re-normalize this cache config field.
- AR-spec CUDA graph sizing uses single-token graphs whenever
  `num_tokens == origin_batch_size` for all methods; full speculative graphs
  still use `batch * (num_spec_tokens + 1)`.
- AR-spec stopping criteria receive batched stop-word tensors. Stop words are
  row-local and `-1` padding must be ignored so one sequence's stop words do
  not stop another sequence.
- `STOP_SESSION` must not reject responses that were already `STOPPED` or
  `TO_BE_MIGRATED` before `scheduler.stop_session()`. When a finish output
  races with a stale cancelled response, `InferOutput.finish` wins and sends
  `ResponseType.FINISH`.
- MP streaming cleanup is part of the runtime contract: session states use
  `init_done`, pre-init cancels return `CANCEL`, `async_end()` cleans state in
  `finally`, and Ray/ZMQ abandoned stream tasks must be dropped with gathered
  output discarded.
- API routes that create engine result generators must close those generators
  and remove sessions on request exit/client disconnect via
  `serve/utils/request_cleanup.py`.
- Engine health probes are background-monitored. Treat `pending` as a skipped
  overlapping probe, use env overrides from `serve/core/health.py`, and keep
  `scheduler_tick` advancing once per forward dispatch.
- GatedDelta preprocessing is an implementation API:
  `GatedDeltaRuleImpl.prepare_inputs()` prepares q/k/g/beta, CUDA can fuse it
  through `kernels/cuda/gated_delta_preprocess.py`, and callers must honor the
  returned `qk_l2norm_done` flag. CUDA fused preprocess supports non-power-of-2
  `kv_ratio` by using a power-of-2 Triton block ratio with masks.
- Guided decoding is wired through normal `BaseModelAgent` sampling. Spec
  decoding currently builds `FusedLogitsProcessor` without a guided manager, so
  do not assume `response_format` works there without redesign and tests.
- Guided decoding CPU-heavy xgrammar operations run through
  `asyncio.to_thread`; keep bitmap fill and accepted-token updates off the
  asyncio event loop.
- Chat-completion `return_logprob` is LMDeploy raw
  `(logprob, token_id)` output and is distinct from OpenAI-compatible
  `logprobs`/`top_logprobs`. It still requires `logprobs_mode` and requests one
  logprob internally.
- `GenerationConfig.return_ppl` asks engines to return summed input
  cross-entropy (`ce_loss`) before sampling; `/get_ppl` accepts raw text or
  token ids and reports the mean prompt cross-entropy score in the `ppl` field.
- Disaggregated weight update has PyTorch-only OpenAI server endpoints:
  `/init_weights_update_group`, `/update_weights_from_distributed`, and
  `/destroy_weights_update_group`. Calls are serialized by an engine lock,
  use a custom external NCCL process group, split `mtp.` weights for
  `qwen3_5_mtp`, and reset CUDA graphs after finished updates.
