# LMDeploy PyTorch Engine Inference Pipeline

End-to-end flow of a single forward step, from user request to token output.

## High-Level Architecture

```
[Pipeline API]
    │
    ▼
[AsyncEngine] ──► [Engine]
    │
    ▼
[EngineLoop] (asyncio event loop)
    │
    ├── main_loop()          ← orchestrates prefill / decode steps
    ├── preprocess_loop()    ← tokenizes and queues requests
    └── send_response_loop() ← streams tokens back to user
    │
    ▼
[InputsMakerAsync]
    │  • decide prefill vs decode (do_prefill_default)
    │  • construct ModelInputs (input_ids, position_ids, block_offsets, ...)
    │  • handle long-context chunking, prefill admission, swap in/out maps
    │
    ▼
[Executor] (RayExecutor / MpExecutor / UniExecutor)
    │  • distributes inputs to workers via RPC (ray.put + forward_async)
    │
    ▼
[RayWorkerWrapper] (one per GPU)
    │
    ▼
[ModelAgent] (one per worker)
    │  ┌──────────────────────────────────────────────────────────────┐
    │  │ _async_loop_inputs_preprocess()                              │
    │  │   • H2D copy on out_stream                                   │
    │  │   • record H2D event and attach _H2DTransfer                 │
    │  │                                                              │
    │  │ _async_loop_background()                                     │
    │  │   • receives preprocessed inputs                             │
    │  │   • self.stream waits on the H2D event                       │
    │  │   • calls _async_step()                                      │
    │  └──────────────────────────────────────────────────────────────┘
    │
    ▼
model_forward()  [agent.py ~L147]
    │  1. ctx_mgr.build_context(inputs, kv/state caches)  →  StepContext
    │  2. model.update_model_metas()
    │  3. model.prepare_inputs_for_generation()  →  input_dict
    │  4. model(**input_dict)  →  hidden_states / output dict
    │
    ▼
[CUDAGraphRunner]  (backends/cuda/graph_runner.py)
    │  ├─ eager path:  model(**kwargs)
    │  └─ graph path:  runner.capture() or runner.forward()
    │
    ▼
Sampling  (FusedLogitsProcessor)
    │
    ▼
_push_output()  →  _out_que
    │
    ▼
get_output_async()  →  D2H copy on out_stream  →  return to EngineLoop
```

## Key Data Structures

### ModelInputs
Produced by `InputsMakerAsync`, consumed by `model_forward()`.

| Field | Shape | Meaning |
|-------|-------|---------|
| `input_ids` | `(1, num_tokens)` | Flattened token ids across batch |
| `seq_length` | `(batch_size,)` | Query length per sequence |
| `history_lengths` | `(batch_size,)` | Cumulative KV length before this step |
| `block_offsets` | `(batch_size, max_blocks)` | Physical block ids in paged KV cache |
| `is_decoding` | bool | True if all sequences have `seq_length==1` |
| `state_offsets` | `(batch_size,)` or `None` | State cache slot ids (SSM models) |
| `is_dummy` | bool | Warmup / padding forward |
| `is_chunk` | bool | Current long-context chunk marker |
| `is_first_chunk` / `is_last_chunk` | bool | Cross-chunk lifecycle markers |
| `is_chunk_multimodal` | bool | Chunk contains multimodal context that models may need to inspect |

### StepContext
Produced by `build_context()`, passed into model via `ctx_mgr.context(context)`.

| Field | Source | Used By |
|-------|--------|---------|
| `q_seqlens` | `inputs.seq_length` | Attention backends |
| `kv_seqlens` | `q_seqlens + history_lengths` | Attention backends |
| `block_offsets` | `inputs.block_offsets` | KV cache read/write |
| `position_ids` | `history_lengths` for decode | RoPE |
| `is_decoding` | `inputs.is_decoding` | Model routing (prefill vs decode) |
| `kv_caches` | `cache_engine.gpu_cache` | Paged KV/custom cache tensors |
| `state_caches` | `state_cache_engine.state_caches` | SSM state buffers |
| `state_offsets` | `inputs.state_offsets` | SSM state slot ids |

## Step Lifecycle inside ModelAgent

```
_async_step()
    │
    ├── if inputs is None (decode step):
    │       apply delta (index_select / reorder running sequences)
    │
    ├── if dp > 1:
    │       _prepare_dp_v1()  (pad batch, broadcast metadata)
    │
    ├── cache_swapping()      (swap in/out blocks between CPU/GPU)
    │
    ├── _async_model_forward()
    │       async_forward() ──► model_forward()
    │       get_logits()
    │
    ├── _slice_outs()         (extract last-token logits for decode)
    │
    └── if need_output:
            _step_postprocess_with_output()
                ├── async_sampling_logits()   (temperature, top-p, top-k, ...)
                ├── stopping_criteria.step()  (eos, max_len, stop words)
                └── _push_output()
```

## Prefix And Chunk Notes

- Prefix caching is scheduler-owned: `Scheduler._schedule_prefill()` calls
  `BlockTrie.match()` before allocation and `BlockTrie.allocate()` after
  allocation. Model inputs then see only the uncached suffix through
  `SchedulerSequence.num_history_ids` and `num_token_ids`. PyTorch VLM/SSM
  have scoped support, AR-spec/MTP uses a private recompute overlap, and
  sliding-window/PD paths still disable the feature.
- Long-context chunking is input-maker/model-agent-owned: non-last chunks advance
  `SchedulerSequence.set_step()` and carry `model_metas` through
  `_prev_chunk_output`, but they do not call `StepInputs.merge_prefill()` or send
  user-visible outputs. They are not computed together with decode steps. Active
  long chunks are engine-local runnable work and can be scheduled even when the
  scheduler queues alone do not show unfinished work.
- Prefix caching and long-context chunking are separate paths; avoid adding a
  shared lifecycle unless a design explicitly calls for it.
- Attention metadata must match the sequence counters: `history_lengths`,
  `q_seqlens`, `kv_seqlens`, `block_offsets`, and `sum_kv_seqlen` define where
  fill kernels write new KV and where prefill attention reads cached KV.

## Streams and Synchronization

| Stream | Purpose | Sync Points |
|--------|---------|-------------|
| `self.stream` | Model forward + sampling | Output event recorded before D2H |
| `out_stream` | H2D (inputs) + D2H (outputs) | H2D event recorded into `_H2DTransfer`, then `self.stream.wait_event()`; output `event.wait()` before D2H |
| `cache_stream` | Cache swap (CPU ↔ GPU) | `cache_engine.events.wait()` before reusing swapped blocks |

**Important**: asyncio is single-threaded but CUDA streams are asynchronous. Never assume ordering between `self.stream` and `out_stream` without explicit events or synchronization, and keep H2D source refs alive until their event completes.
