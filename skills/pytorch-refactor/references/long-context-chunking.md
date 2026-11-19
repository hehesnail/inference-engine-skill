# Long-Context Chunking

Use this when changing current long-context chunking, chunk flags, multimodal
chunk slicing, or the model-agent handoff between chunks. This is separate from
prefix caching.

## Current Scope

- Implementation owner: `engine/inputs_maker.py::LongContextChunker` plus
  scheduler reservation policy and model-agent chunk state in
  `engine/model_agent/agent.py`.
- It chunks one oversized prefill request across multiple forward passes when
  `seq.num_token_ids > max_prefill_token_num`.
- It is not official chunked prefill: chunks are not computed together with
  decode steps, and non-last chunks do not enter persistent decode `StepInputs`.
- Current opt-TTFT behavior is first-slice/eager long-prefill chunking. Long
  chunks are still separate forwards, but decode and bounded short-prefill
  turns are preferred before continuing non-final long chunks.
- `MiscConfig.enable_chunked_prefill` and
  `InputsMakerAsync.do_prefill_chunked()` exist as an internal/unwired path.
  `MiscConfig.from_engine_config()` currently hard-codes
  `enable_chunked_prefill=False`.
- `max_prefill_token_num` both caps scheduler prefill admission and acts as the
  chunk size budget; `ExecutorBase` may reduce it during memory sizing.
- TTFT policy envs are read in `envs.py`: `LMDEPLOY_PT_TTFT_POLICY`
  (`size` or `fifo`), `LMDEPLOY_PT_TTFT_SHORT_TURNS`, and
  `LMDEPLOY_PT_TTFT_AGING_SEC`.

## Files

- `lmdeploy/pytorch/engine/inputs_maker.py`: `LongContextChunker`,
  `create_model_inputs_long_context()`, prefill/decode admission.
- `lmdeploy/pytorch/paging/scheduler.py`: `schedule(...,
  allow_long_prefill=..., prefer_long_prefill=...)`,
  `reserve_long_context_chunk()`, and `schedule_running()`.
- `lmdeploy/pytorch/engine/model_agent/agent.py`: `_prev_chunk_output`,
  `_prepare_inputs_prefill()`, non-last chunk handling.
- `lmdeploy/pytorch/model_inputs.py`: chunk flags and `StepContext` derivation.
- `lmdeploy/pytorch/spec_decode/spec_agent.py`: carries target hidden states,
  position ids, target input embeds, and mrope state across chunk boundaries;
  suppresses real draft sampling for non-last chunks.
- `lmdeploy/pytorch/backends/*/op_backend.py`: attention metadata from
  `q_seqlens`, `kv_seqlens`, `history_lengths`, and `block_offsets`.

## Flow

1. `InputsMakerAsync.do_prefill_default()` decides prefill vs decode, forcing
   the final pending long chunk and otherwise honoring the normal prefill
   interval/token/batch heuristics.
2. `_make_forward_inputs()` applies the opt-TTFT policy: short/normal prefill
   can run while long work waits, long-work turns can prefer one long waiter,
   and active non-final chunks usually yield one loop after they run.
3. Scheduler prefill admission uses `allow_long_prefill` for short turns and
   `prefer_long_prefill` for long-work turns. Waiting long prompts may be
   ordered by size with age credit, or FIFO.
4. If exactly one scheduled prefill sequence exceeds `max_prefill_token_num`,
   `LongContextChunker.set_seq(seq)` starts chunk mode.
5. `LongContextChunker.next_chunk_size()` returns the text chunk size and
   optional multimodal payload. It avoids splitting a multimodal span and may
   enlarge the budget to fit the largest multimodal item.
6. `create_model_inputs_long_context()` builds `ModelInputs` for the current
   chunk and marks `is_chunk=True`.
7. Non-last chunks have `delta=None`; `update_running_seqs()` advances the
   chunker with `seq.set_step(next_step)` and does not schedule decode work with
   that chunk.
8. `EngineLoop._make_infer_outputs()` suppresses user outputs for non-last
   chunks and accumulates logits/routed experts on the sequence.
9. `BaseModelAgent._prepare_inputs_prefill()` carries `model_metas` from
   `_prev_chunk_output` into the next chunk.
10. `_async_step()` stores `_prev_chunk_output` for non-last chunks and calls
   `step_inputs.merge_prefill()` only for the final chunk or normal prefill.
11. The last chunk uses normal prefill input construction for the remaining
    suffix, marks `is_last_chunk=True`, clears the chunker, and merges into
    persistent decode state.

## Metadata and Attention

- Chunk flags on `ModelInputs`: `is_chunk`, `is_first_chunk`, `is_last_chunk`,
  `is_chunk_multimodal`.
- `StepContext.new()` derives `q_seqlens`, `kv_seqlens`, `position_ids`, and
  `block_offsets` from `ModelInputs` counters.
- CUDA prefill metadata sets `kv_start_loc`, `kv_flatten_size=sum_kv_seqlen`,
  `cu_seqlens_q`, and `cu_seqlens_k`.
- Dlinfer computes `is_prefill_no_cache` from `q_seqlens == kv_seqlens`; a
  long-context chunk with history is a cached prefill path, not no-cache prefill.
- Gated-delta/SSM backends use `state_offsets` from `ModelInputs`/`StepContext`;
  CUDAGraph pads these as `state_ids`.

## Design Checks

- Preserve sequence counters: completed chunks must move from `num_token_ids`
  into `num_history_ids` before the next model input.
- Do not call `StepInputs.merge_prefill()` or emit user-visible outputs for
  non-last chunks.
- Active long chunks are engine-local runnable work. `EngineLoop` must check
  `InputsMakerAsync.has_pending_long_context_chunk()` in addition to
  `scheduler.has_unfinished()`.
- Prefix-cache hits can shrink a long prompt enough that the tail is a normal
  prefill or starts from a middle chunk. Do not roll back a valid hit solely
  because it resumes a multi-chunk prompt.
- Check all paradigms (`ar`, `ar_spec`, `dllm`) when adding/changing chunk flags
  or `ModelInputs` fields.
- Check VLM/multimodal behavior when changing chunk sizes or slicing; chunking
  avoids splitting multimodal spans.
- Treat `enable_chunked_prefill` as a separate unwired path unless the task is
  explicitly about implementing that scheduler mode.

## Validation

- Unit: input-maker tests for `LongContextChunker`, chunk flags, multimodal
  boundaries, opt-TTFT turn selection, active chunk deferral, and
  `seq.set_step()` progression.
- Unit: scheduler tests for `allow_long_prefill`, `prefer_long_prefill`,
  `reserve_long_context_chunk()`, and decode-growth allocation under KV
  pressure.
- Unit: model-agent tests for `_prev_chunk_output`, non-last output suppression,
  and final `merge_prefill()`.
- Integration: one long prompt exceeding `max_prefill_token_num`, then decode a
  few tokens and compare against non-chunked behavior on a small model.
- Optional: cover `ar_spec` and one VLM path when chunk flags, model metas, or
  multimodal slicing change.
- For spec decode, include a long-context chunk case that crosses first/middle/
  last chunk boundaries when changing target hidden state, target input embeds,
  position ids, or mrope logic.
