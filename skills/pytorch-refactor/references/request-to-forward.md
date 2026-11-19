# LMDeploy PyTorch Request-to-Forward Trace

Practical, evidence-driven trace for how LMDeploy PyTorch backend turns requests into tokens.
Use the phase headings as a menu; read only the phase relevant to the current
change unless debugging an end-to-end lifecycle bug.

## Phase 1: Engine Bootstrapping

Primary file: `lmdeploy/pytorch/engine/engine.py`

1. `Engine.__init__` normalizes and validates runtime config.
2. `ConfigBuilder` creates `scheduler_config`, `cache_config`, `backend_config`, `dist_config`, `misc_config`.
   If speculative decoding is enabled, `SpecDecodeConfig` is built with a
   draft `DistConfig`; `qwen3_5_mtp` clones the main dist config, while other
   spec methods use a default draft config today.
3. `build_executor(...)` constructs executor (`uni` / `mp` / `ray`) and initializes model workers.
4. Strategy factory is built from model paradigm (`ar` / `dllm` / `ar_spec`).
5. Scheduler and request manager are created and main async loop is bound.

Refactor anchors:
- Keep config build order stable unless there is a strong reason.
- Any change in executor selection should preserve distributed invariants.

## Phase 2: Executor Selection

Primary file: `lmdeploy/pytorch/engine/executor/__init__.py`

- `get_distributed_executor_backend(world_size, dp, device_type)` picks backend.
- Current policy:
  - `world_size == 1` -> `uni`
  - `dp > 1` -> `ray` (hard assertion)
  - otherwise device capability may pick `ray` or `mp`

Refactor anchors:
- Preserve the explicit `dp > 1` -> `ray` semantic contract unless requirement changes.
- Keep backend selection deterministic and log-visible.

## Phase 3: Model Rewrite Resolution

Primary files:
- `lmdeploy/pytorch/models/module_map.py`
- `lmdeploy/pytorch/models/patch.py`

Flow:
1. Architecture name is read from HF config (`architectures` / `auto_map`).
2. `_get_module_map()` merges default map, device-special map, and custom map.
3. Rewrite class is resolved and instantiated through `build_patched_model()`.
4. Optional LoRA adapters are injected via `add_adapters()`.

Refactor anchors:
- Mapping hit priority is behavior-critical.
- Device-specific map overlays must stay explicit.

## Phase 4: Input Making and Forward Preparation

Primary files:
- `lmdeploy/pytorch/engine/inputs_maker.py`
- `lmdeploy/pytorch/engine/model_agent/inputs_maker.py`
- `lmdeploy/pytorch/engine/model_agent/agent.py`

Flow:
1. `InputsMakerAsync` decides prefill vs decode, calls the scheduler, builds
   `ModelInputs`/`ModelInputsDelta`, handles opt-TTFT long-context chunks, and
   prepares sampling/stopping/extra inputs.
2. Executor sends the forward-input dict to workers.
3. `BaseModelAgent._async_loop_inputs_preprocess()` moves input payloads to GPU.
4. Worker-side `engine/model_agent/inputs_maker.py` is used for DP/dummy input
   coordination inside the model agent.
5. `BaseModelAgent._async_step()` resolves prefill vs decode state and drives
   cache swapping, model forward, sampling, and output enqueue.

DP worker coordination:
- `DPForwardInputsMaker` inserts dummy inputs on ranks without local work and
  creates dummy `ExtraInputs` when the strategy needs them.
- `_prepare_dp_v1()` all-gathers named scalar metadata from
  `engine/model_agent/dp_utils.py`: local `is_decoding`, `is_dummy`,
  `num_tokens`, `is_sleeping`, `batch_size`, optional `draft_num_tokens`, and
  optional `enable_microbatch`.
- If any DP rank is prefill, all ranks run the step as prefill; the original
  local decode flag is preserved as `dp_meta.is_decoding`, while the gathered
  global flag is `dp_meta.dp_is_decoding`.

Refactor anchors:
- Preserve synchronization semantics for distributed input availability.
- Keep dummy-input generation isolated from real-request path assumptions.
- Active long-context chunks are local to `InputsMakerAsync`; engine-loop
  runnable checks must include `has_pending_long_context_chunk()` instead of
  relying only on scheduler queues.
- `scheduler.tick()` marks one forward dispatch and is called after
  `executor.forward_async()` in `InputsMakerAsync._send_next_inputs_impl()`.
  Health monitoring relies on this tick; do not move it back to response
  collection.

## Phase 4.5: StepInputs Lifecycle (Prefill -> Decode Transition)

Primary files:
- `lmdeploy/pytorch/engine/model_agent/agent.py` (`BaseModelAgent._async_step()`)
- `lmdeploy/pytorch/strategies/base/step_inputs.py`
- `lmdeploy/pytorch/strategies/{ar,dllm,ar_spec}/step_inputs.py`

The engine maintains a persistent `StepInputs` object that holds the running
decode batch across steps. Two input paths exist.

### Path A: Prefill (full inputs present)

```
InputsMaker.create_model_inputs() -> fresh ModelInputs from scheduler sequences
  -> model_agent_strategy.make_extra_inputs() -> ExtraInputs
  -> model_agent_strategy.make_stopping_criteria() -> StoppingCriteria
  -> BaseModelAgent._prepare_inputs_prefill()
       if delta is not None: StepInputs.reindex(delta)
       if chunk continuation: carry model_metas from _prev_chunk_output
  -> model forward
  -> sampling
  -> StepInputs.merge_prefill():
      1. create first decode-step ModelInputs
         -> AR: get_model_inputs_next_decoding()
         -> DLLM: _get_model_inputs_next_decoding(), with block-size step_seqlens
         -> AR_Spec: AR conversion plus next token + draft tokens
      2. advance SamplingInputsDelta with paradigm helper
      3. If first merge: assign directly; else concatenate into decode batch
         -> merge_model_inputs()
         -> extra_inputs.merge()
         -> stopping_criteria.merge()
```

Non-last long-context chunks are special: `_async_step()` stores
`_prev_chunk_output` and does not call `merge_prefill()` until the final chunk.
They are not computed together with decode steps; this is the current
long-context chunker, not the official chunked-prefill mode.

### Path B: Decode (delta inputs only)

```
InputsMaker.create_model_inputs_delta() -> ModelInputsDelta
  -> BaseModelAgent._get_inputs_from_delta()
       StepInputs.reindex(delta)
       sampling_inputs.update_delta(step_inputs.sampling_delta)
  -> model forward
  -> sampling
  -> StepInputs.step_decode():
      1. update next-step ModelInputs
         -> AR: model_inputs.step(next_token_ids)
         -> DLLM: _update_dllm(), then model_inputs.step(..., step_seqlens)
         -> AR_Spec: rejected-token-aware step_seqlens, then step()
      2. replace/advance extra_inputs, stopping_criteria, and sampling_delta
```

### ModelInputsDelta is NOT a tensor diff
`ModelInputsDelta` carries an **index-select specification**, not deltas of tensor values:
- `indices`: which batch elements survive (after eviction/completion)
- `block_offsets`: fresh block table from scheduler
- `max_kv_seqlen`, `sum_kv_seqlen`: recomputed aggregates
- `num_ignored_history`: updated sliding window offsets

### Key invariant
`input_ids` shape varies by paradigm during decode:
- AR: `(1, batch_size)` — one token per sequence
- DLLM: `(1, batch_size * block_size)` — one block per sequence
- AR_Spec: `(1, batch_size * (1 + num_spec_tokens))` — verified + draft tokens per sequence

DLLM and AR_Spec must **reshape** input_ids before index_select, then flatten back.

Refactor anchors:
- `StepInputs.reindex()/merge_prefill()/step_decode()` form the core lifecycle;
  changes here affect all paradigms.
- Paradigm-specific hooks live in `strategies/{ar,dllm,ar_spec}/step_inputs.py`
  and shared AR helpers in `strategies/ar/model_inputs.py`.
- Adding a new field to `ModelInputs` requires updating:
  `merge_model_inputs()`, AR/DLLM/ARSpec reindex helpers,
  prefill-to-decode conversion helpers, `ModelInputs.step()` if needed, and
  dummy input creation.
- Prefix-cache SSM state fields are compact host offset sequences:
  `state_prefix_cache_offsets`, `state_prefix_cache_dst_offsets`,
  `state_prefix_cache_save_src_offsets`, and
  `state_prefix_cache_save_offsets`. Keep merge/reindex/prefill-to-decode
  helpers preserving them without CUDA boolean-mask/nonzero work.

## Phase 4.6: Spec Decode + DP/MTP Contract

Primary files:
- `lmdeploy/pytorch/spec_decode/{__init__,base,spec_agent}.py`
- `lmdeploy/pytorch/strategies/ar_spec/`
- `lmdeploy/pytorch/engine/model_agent/{agent,dp_utils,inputs_maker}.py`
- `lmdeploy/pytorch/engine/executor/base.py`

Spec agent construction:
- `build_spec_agent()` enables a real `SpecModelAgent` on every draft TP rank
  that participates. For main TP > 1 with draft TP = 1, only rank 0 of each
  main TP group runs the draft model; other ranks get `BaseSpecModelAgent`.
- `BaseSpecModelAgent` still carries config, strategy, rejection sampler, and
  draft dist context even when it is a no-op wrapper.

Main step ordering:
1. Main model forward returns target hidden states/logits.
2. `spec_agent.async_sampling_logits()` runs target logits processing and
   rejection sampling, producing `next_token_ids`, `output_token_ids`, and
   logprobs.
3. TP ranks broadcast the next token and pre-draft extra fields.
4. `spec_agent.async_model_forward()` converts main inputs to draft inputs and
   runs draft forwards.
5. `spec_agent.post_broadcast()` broadcasts draft token ids needed by
   `ARSpecExtraOutputs`.

DP/MTP details:
- `_prepare_dp_v1()` computes `dp_meta.dp_draft_num_tokens` because MTP draft
  inputs are shifted relative to main inputs. First long-context chunks drop
  one token per sequence; last chunks append one token per sequence.
- `SpecModelAgent._prepare_inputs_from_main()` rebuilds draft `DPMeta` under
  the draft dist context when `dp_draft_num_tokens` is present.
- Local decode state and DP-global decode state are distinct. Use
  `ModelInputs.is_decoding` for local rank behavior, and
  `ModelInputs.global_is_decoding()` / `StepContext.global_is_decoding()` for
  gathered DP state such as padding, graph keys, and global decode checks.
- Do not overwrite `ModelInputs.is_decoding` with the gathered DP state in
  mixed prefill/decode batches. Draft MTP inputs preserve local `is_decoding`;
  padding and graph decisions use `global_is_decoding()`.
- DP=1 non-last chunks can skip the remaining speculative forward loop and
  emit dummy draft ids. DP>1 non-last chunks must still run all draft forwards
  so all ranks stay synchronized.
- Long-context chunk index shifts are part of the draft contract:
  first chunks decrement `last_token_indices`, last chunks increment them after
  prepending/appending carried state.
- Draft cache sizing uses `SpecDecodeConfig.dist_config.attn_tp`; see
  `references/cache-engine.md`.
- Prefix cache is allowed on AR-spec/MTP through a one-block target recompute
  overlap: `match_recompute_blocks=1` drops the overlap from the public hit and
  keeps the overlap blocks private during trie allocation.
- AR-spec stopping criteria compare batched stop words per row and ignore `-1`
  padding. Keep stop-word tensors row-local so a stop token configured for one
  sequence does not stop another sequence in the same batch.

## Phase 4.8: Async Pipeline and CUDA Streams

Primary file: `lmdeploy/pytorch/engine/model_agent/agent.py`

The model agent runs two async coroutines on the **same single-threaded asyncio event loop**, each with its own CUDA stream:

- `_async_loop_inputs_preprocess`: H2D copies on `self.out_stream`, records a
  CUDA event, attaches `_H2DTransfer`, and puts the dict to `_in_que`
- `_async_loop_background`: pops `_H2DTransfer`, keeps source refs alive, waits
  on the event from `self.stream`, then runs model forward

Since asyncio is single-threaded cooperative multitasking, the two coroutines never run concurrently on the CPU. They only switch at `await` points. Pipelining H2D with forward is only beneficial if the next batch is ready in `_pre_in_que` before the current forward finishes.

Stream synchronization: H2D completion is represented by an event recorded on
`out_stream`; the forward stream waits with `self.stream.wait_event(...)`.
`_pending_h2d_transfers` keeps CPU-side source refs alive until the async copy
event completes, including queue-drain paths during sleep.

Refactor anchors:
- The two streams (`out_stream` for H2D, `self.stream` for forward) require explicit synchronization — they have no implicit ordering.
- Do not drop `_H2DTransfer` refs before the event completes; non-blocking H2D
  copies may still be reading CPU-side objects.
- Do not assume CPU concurrency between the two coroutines — asyncio is single-threaded.

## Phase 5: Forward Execution

Primary function: `model_forward(...)` in `agent.py`

Core sequence:
1. Enter CUDA stream and `step_ctx_manager(model.ctx_mgr)`.
2. Build context via `ctx_mgr.build_context(...)` with KV caches and state caches.
3. Enter `ctx_mgr.context(context)`.
4. Update model metas (`update_model_metas`).
5. Build generation inputs (`prepare_inputs_for_generation`).
6. Run model forward and normalize output dict.
7. Return `model_metas`, sequence lengths, and position ids for downstream usage.

Refactor anchors:
- Context lifecycle order is critical; do not reorder casually.
- Current main has no `StateCacheEngine.init_caches()` call; SSM state cache
  access flows through `state_caches` and `state_offsets` in `StepContext`.
- SSM prefix-cache restore/save copies happen inside `model_forward()` on the
  forward stream. Restore copies `checkpoint -> runtime_state` before
  `update_model_metas()` and model forward; save copies
  `runtime_state -> checkpoint` after forward and before returning output.
- Keep output fields expected by downstream sampling/spec-decode code.

## Phase 6: Cache and Sampling

Primary concepts in `agent.py` and `engine/logits_process.py`:
- `cache_swapping(...)` may perform `swap_in` / `swap_out`.
- When swap is issued, code waits on `cache_engine.events.wait()`.
- Sampling uses `FusedLogitsProcessor`, `SamplingInputs`, `SamplingInputsDelta`.
- `GenerationConfig.return_ppl` asks the backend to compute prompt CE loss
  before sampling. PyTorch computes this via `compute_input_ce_loss()` and
  chunked prompts carry the previous chunk's last logit for the boundary token.
- For prefix-cache SSM checkpoints, `EngineLoop._publish_forward_prefix_cache()`
  commits reserved checkpoints and releases restore pins after the current
  forward input has been queued, before prefetching the next batch. Save
  producer refs are released after `executor.get_output_async()` reaches the
  output/event boundary.

Refactor anchors:
- Cache event wait is a hard synchronization point.
- Sampling data structures are cross-module contracts.
- Do not add CUDA synchronizes around prefix-cache state copies. Correctness is
  based on forward-stream ordering plus checkpoint restore/save refcounts.

## Phase 7: Response and Stop Races

Primary files:
- `lmdeploy/pytorch/engine/engine.py`
- `lmdeploy/pytorch/engine/engine_loop.py`

Refactor anchors:
- `STOP_SESSION` snapshots responses already in `MessageStatus.STOPPED` or
  `TO_BE_MIGRATED` before calling `scheduler.stop_session()`. Those responses
  should not be rejected afterward; only newly stopped/running responses should
  receive cancel.
- `EngineLoop._send_resp()` treats `InferOutput.finish=True` as authoritative.
  A finish output must send `ResponseType.FINISH` even if the response object
  has already been marked cancelled by a stale stop path.

## Refactor Safety Checks

Before and after edits, verify:
1. Dist invariants (`dp/tp/ep/world_size`) still hold.
2. Cache swap path still waits for completion events.
3. Model rewrite still resolves target architectures.
4. Prefill/decode paths both produce valid outputs.
5. `ar_spec` path still receives valid `specdecode_config`.
6. DP + spec paths preserve `dp_meta` local/global decode flags and
   `dp_draft_num_tokens` through main-to-draft conversion.
7. Stop-session and client-cancel tests still cover already-stopped sequences,
   finish-vs-cancel races, and MP generator cleanup.
