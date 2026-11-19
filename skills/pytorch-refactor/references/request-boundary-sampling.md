# Request Boundary, Sampling, Guided Decoding

Use this when changing request preprocessing, `GenerationConfig` translation,
sampling behavior, guided decoding, logprobs, returned logits, routed expert
outputs, or multimodal request metadata.

## Source Anchors

- External runtime request API:
  - `engine/engine_instance.py::EngineInstance.async_stream_infer`
  - converts `GenerationConfig` to `messages.py::SamplingParam`
  - sends `ADD_SESSION` and `ADD_MESSAGE` with `token_ids`,
    `adapter_name`, `input_multimodals`, migration/cache flags, and sampling
    params
- Request preprocessing:
  - `engine/input_process.py::PreprocessInputResult`
  - `BaseModelInputProcessor.preprocess_input()`
  - model-specific processors usually return `input_ids`, `input_multimodals`,
    and optional `model_metas`
  - VLM frontend preprocessing: `serve/processors/multimodal.py`,
    `vl/media/*`, `vl/model/base.py`, `vl/model/preprocess_utils.py`
- Sequence/session data:
  - `messages.py::SequenceManager`, `SchedulerSession`, `SchedulerSequence`
  - histories: `HistoryTokenIds`, `HistoryEmbeddings`,
    `HistoryMultiModals`, `HistoryMropePosIds`, `HistoryLogits`,
    `HistoryRouterExperts`
  - `multimodal/data_type.py::MultiModalData`
- Sampling construction:
  - `strategies/{ar,dllm,ar_spec}/sampling.py::make_sampling_inputs()`
  - `engine/logits_process.py::SamplingInputs`
  - `engine/logits_process.py::SamplingInputsDelta`
- Logits processing:
  - `engine/model_agent/agent.py::BaseModelAgent.async_sampling_logits()`
  - `engine/logits_process.py::FusedLogitsProcessor`
  - `spec_decode/spec_agent.py` for spec decode expansion/slicing and target
    logits processing
- Guided decoding:
  - `engine/guided_process.py::GuidedDecodingManager`
  - uses xgrammar processors keyed by `session_id` and `seq_id`

## Flow

1. Public callers enter through `EngineInstance.async_stream_infer()` or an
   `MPEngineInstance`.
2. `SamplingParam.from_gen_config()` normalizes generation options such as
   top-k/top-p/min-p, temperature, stop/bad words, logprobs, response format,
   custom logits processors, routed expert return, and `return_ppl` into
   backend flags such as `out_ce_loss`.
3. Scheduler sequences keep token, multimodal, logits, mrope, and routed expert
   history. `HistoryMultiModals.update_multimodals()` shifts multimodal spans by
   the previous valid token length.
4. Strategy-specific `make_sampling_inputs()` batches per-sequence sampling
   params. AR gathers `all_ids` only when custom logits processors need them and
   gathers `generated_ids` for repetition penalty or repetition ngram.
5. `BaseModelAgent.async_sampling_logits()` creates `FusedLogitsProcessor`,
   applies guided/custom/repetition/temperature/bad-word/top-k/top-p/min-p
   processing, samples the next token, and computes optional logprobs.

## Multimodal / VLM Boundary

- Public OpenAI-style multimodal content is normalized in
  `serve/processors/multimodal.py`. Supported typed items include image,
  video, audio, and time series variants.
- Media IO lives under `vl/media/*`. Audio IO lazily imports `librosa` and
  `soundfile`, and defaults to 16 kHz for Qwen3 Omni unless overridden.
- `VisionModel.preprocess()` groups raw items by modality, calls the HF
  processor, maps processor output attributes through
  `ATTR_NAME_TO_MODALITY`, then expands bundled outputs with
  `get_expanded_mm_items()`.
- `get_expanded_mm_items()` must preserve prompt order by sorting expanded
  items by offset. Qwen3-VL video may split into per-frame items, while Qwen3
  Omni video keeps whole-video spans and carries `second_per_grid`.
- Bundled multimodal expansion validates offset cardinality: image offsets must
  match image grids; video offsets must be either per-video or per-frame. Audio
  expansion preserves per-item `feature_attention_mask` slices.
- PyTorch VLM prefix caching depends on stable multimodal identity. When
  effective prefix caching is enabled, populate `MultiModalData.content_hash`
  with `ensure_multimodal_content_hashes()` after model preprocessing so trie
  keys can distinguish identical placeholder tokens backed by different media.
- `MultiModalData` carries `modality`, feature data, `start/end`, optional
  `mrope_pos_ids`, and model-specific `meta` such as image/video/audio token
  ids and audio feature lengths.
- Model-specific input processors own final packing. Qwen3 Omni packs image,
  audio, and video into `mm_data`; audio converts `feature_attention_mask` to
  `audio_feature_lengths` and creates audio MROPE ids over the token span.

## Hard Invariants

- `SamplingInputsDelta` is not a user-facing request object. It is the tensor
  state persisted inside `StepInputs`; adding tensor fields may require updates
  in `SamplingInputs.get_delta()`, `update_delta()`, `to_device()`,
  strategy `step_sampling_inputs()`, reindex/merge helpers, and
  `spec_decode/spec_agent.py` expand/slice helpers.
- Guided decoding supports `json_schema`, `regex_schema`, and `json_object` via
  `response_format`, but normal and spec paths are not equivalent. Current spec
  decode target-logits processing does not pass `GuidedDecodingManager`.
- Custom logits processors require `sampling_inputs.all_ids`; do not remove the
  gather in strategies unless processors are impossible for that path.
- `FusedLogitsProcessor` may wait once for the current CUDA stream before CPU or
  Python-side processors. Keep async stream behavior explicit.
- Guided decoding xgrammar work is CPU-heavy. `FusedLogitsProcessor` fills the
  guided bitmask and accepts sampled guided tokens through `asyncio.to_thread`;
  keep those operations off the main asyncio event loop when refactoring.
- Normal sampling calls `accept_guided_tokens()` after token sampling. Spec
  decoding still does not pass a `GuidedDecodingManager` to its target-logits
  `FusedLogitsProcessor`, so guided decoding behavior is not automatically
  shared between normal and spec paths.
- AR-spec stop words are batched per row. Compare each row only with its own
  stop words and ignore `-1` padding; otherwise different stop-word settings in
  a batch can leak across sequences.
- `MultiModalData.start/end` are prompt positions. When appending new request
  multimodals to history, offset them by existing valid tokens.
- MROPE multimodal data needs `mrope_pos_ids` when `use_mrope` is true, except
  modality-specific paths that intentionally skip it.
- For models that scatter multimodal embeddings by token id, update
  `DeployModelMixinV1.get_multimodal_mask()` for every supported modality.
- Chat-completion `return_logprob` requests raw `(logprob, token_id)` pairs in
  `output_token_logprobs`. It is separate from OpenAI-compatible `logprobs` and
  `top_logprobs`, but still requires `logprobs_mode` and internally asks the
  engine for one logprob.
- `return_ppl` is prompt scoring, not generation scoring. PyTorch computes
  summed input CE loss before sampling and skips decode steps; chunked prompts
  must carry the previous chunk's last logit to score the cross-chunk boundary.
  The async engine rejects PPL scoring when prefix caching or speculative
  decoding is enabled.
- Routed expert history and returned logits are sequence-history features, not
  model-input fields. Check `SchedulerSequence.append_routed_experts()` and
  `append_logits()` before changing output shapes.
- Prefix-cache routed-expert replay is best-effort sequence history replay.
  Routed experts are not part of cache identity; missing prefix rows must not
  block KV/state reuse or shift later routed-expert history.

## Validation

- Sampling/logits: `tests/pytorch/engine/test_logits_process.py`
- Spec sampling: `tests/pytorch/spec_decode/test_reject_sample.py`,
  `tests/pytorch/spec_decode/test_spec_agent.py`,
  `tests/pytorch/spec_decode/test_strategies.py`
- Runtime streaming/cancel side effects: `tests/pytorch/engine/test_request.py`,
  `tests/pytorch/engine/test_engine_sleep.py`
- VLM/audio preprocessing: `tests/test_lmdeploy/test_vl/test_qwen3_omni_processor.py`,
  `tests/test_lmdeploy/test_vl/test_preprocess_utils.py`,
  `tests/test_lmdeploy/test_content_merge.py`
