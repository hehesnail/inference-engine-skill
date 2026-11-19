# CUDAGraph Support Rules for LMDeploy Models

Rules that every model must obey if it wants `support_cuda_graph()` to return `True` for decode steps.

## Why CUDAGraph Matters

Decode steps are small (batch_size × 1 token) but launch hundreds of kernels. CUDA Graph eliminates per-kernel CPU launch overhead by recording the entire GPU command sequence once and replaying it. This typically yields **10–30 % decode throughput improvement**.

## The Three Hard Rules

### Rule 1: No CUDA Synchronize Inside `model(**input_dict)`

The code captured by `torch.cuda.graph()` must be a pure GPU compute stream.

**Forbidden operations inside the captured region:**

- `cudaStreamSynchronize` / `cudaDeviceSynchronize`
- `torch.cuda.synchronize()`
- Any operation that implicitly blocks the host until GPU work finishes

**Where this is enforced:**

```python
# agent.py ~L147
def model_forward(model, inputs, cache_engine, state_cache_engine, stream):
    with torch.cuda.stream(stream), step_ctx_manager(model.ctx_mgr):
        context = ctx_mgr.build_context(inputs, ...)

        with ctx_mgr.context(context):
            model_metas = model.update_model_metas(...)
            input_dict = model.prepare_inputs_for_generation(...)

            # CAPTURE BOUNDARY when CUDA graph runner wraps the model:
            # GraphRunner.capture()/forward() calls the underlying model.
            output = model(**input_dict)          # ← must be pure GPU
```

If a sync sneaks into `model(**input_dict)`, capture fails with:

```
torch.AcceleratorError: CUDA error: operation not permitted when stream is capturing
```

### Rule 2: No Device → Host Memory Read Inside the Captured Region

Any operation that copies a GPU scalar value back to Python land implicitly triggers a stream sync.

**Forbidden inside `model(**input_dict)`:**

| Operation | Why it breaks graph |
|-----------|---------------------|
| `tensor.item()` | Host must wait for GPU to produce the scalar |
| `tensor.tolist()` | Same, for multiple scalars |
| `tensor.cpu()` | D2H copy blocks host |
| `tensor.numpy()` | D2H copy + numpy conversion |
| `int(tensor)` / `float(tensor)` | Implicit `.item()` |
| `print(tensor)` | Implicit `.item()` for formatting |
| `if tensor.any():` | Forces sync to evaluate boolean on host |
| `for i in range(tensor.size(0)):` where size depends on runtime value | Dynamic loop bounds from GPU data |

**Workaround**: Move the read **outside** `model()`, or replace the scalar value with a **fixed-size tensor + mask**.

### Rule 3: Fixed Intermediate Tensor Shapes

Every tensor created inside `model(**input_dict)` must have the **same shape at capture time and at every replay**.

**Allowed:**
- `torch.zeros(batch_size, fixed_dim)` where `batch_size` is padded to capture size
- `torch.where(mask, a, b)` — shape is fixed, only values change
- Indexing with fixed indices: `cache[layer_id, block_ids, offsets]`

**Forbidden:**
- `torch.arange(dynamic_len)` where `dynamic_len` changes per step
- `tensor[:, :runtime_value]` where `runtime_value` comes from GPU data
- `torch.cat([a, b], dim=0)` where the concatenation length varies

**Workaround**: Allocate the **maximum possible size** and mask unused entries.

```python
# ❌ BROKEN: shape grows with history_length
num_compressed = total_lens // ratio
max_compressed = int(num_compressed.max().item())
compressed_kv = torch.zeros(batch_size, max_compressed, head_dim)

# ✅ FIXED: allocate max possible, mask unused
MAX_COMPRESSED = max_history_len // ratio
compressed_kv = torch.zeros(batch_size, MAX_COMPRESSED, head_dim)
mask = torch.arange(MAX_COMPRESSED) < num_compressed.unsqueeze(1)
compressed_kv = torch.where(mask.unsqueeze(-1), compressed_kv, 0)
```

## CUDAGraphRunner Architecture

```
CUDAGraphRunner.__call__(**kwargs)
    │
    ├── enable_graph = model.support_cuda_graph(**kwargs)
    │
    ├── if not enable_graph:
    │       return model(**kwargs)               # eager path
    │
    ├── graph_key = (padded_batch_size, is_decoding, microbatch, query_len)
    │
    ├── if graph_key not in _runner_map:
    │       runner = CUDASingleGraphRunner(...)
    │       output = runner.capture(**kwargs)     # warmup + capture
    │       _runner_map[graph_key] = runner
    │       return output
    │
    └── else:
            runner = _runner_map[graph_key]
            return runner.forward(**kwargs)       # fill buffers + replay
```

### Buffer Management (CudaGraphMixin)

| Method | Called When | Purpose |
|--------|-------------|---------|
| `make_buffers_cudagraph(meta, ...)` | `runner.capture()` (once) | Create fixed-size input buffers |
| `fill_buffers_cudagraph(meta, ...)` | Every `capture()` and `forward()` | Copy real inputs into buffer front slots |
| `update_context_cudagraph(meta, ctx)` | Every `capture()` and `forward()` | Redirect `ctx.*` pointers to buffer tensors |
| `get_outputs_cudagraph(buffers, ...)` | Every `capture()` and `forward()` | Slice buffer outputs to actual token count |

Current graph-key notes:
- Spec/decode graph keys distinguish query length, not draft-vs-target hidden
  width. The key is `(capture_batch, global_is_decoding, enable_microbatch,
  query_len)`, where `query_len = input_ids.size(1) // batch_size`.
- `CudaGraphMeta.decode_query_len` is captured with the runner and used when
  creating/filling FlashMLA metadata and FA3 scheduler metadata. Do not infer
  decode query length from padded `max_tokens // max_batchs`.
- Graph enablement and graph keys use `StepContext.global_is_decoding()`, not
  only local `is_decoding`, so DP mixed prefill/decode batches do not capture a
  decode graph on ranks that are globally in prefill mode.
- In DP decoding, `CUDAGraphRunner.update_inputs()` syncs `dp_meta` TP sizes as
  padded capture batch size times query length, so spec decode with
  `num_spec_tokens + 1` remains shape-stable.
- `PytorchEngineConfig.cudagraph_capture_batch_sizes` / `CacheConfig` may
  restrict capture sizes. Normalize configured sizes to sorted unique positive
  integers no larger than `max_batch_size`, always append `max_batch_size`, and
  let graph runners defensively re-normalize before returning capture sizes.
- For AR-spec CUDA graph sizing, all methods use a single-token graph when
  `num_tokens == origin_batch_size`. Full speculative captures still allocate
  `batch * (num_spec_tokens + 1)`.
- `LMDEPLOY_FAKE_CUDA_GRAPH_CAPTURE=1` disables actual CUDA graph replay while
  still exercising graph padding and buffer-filling behavior. This is useful
  for focused graph-shape tests.

### Capture vs Replay Flow

```python
# CAPTURE (first occurrence of a graph_key)
def capture(self, **kwargs):
    self.meta.input_buffers = model.make_buffers_cudagraph(self.meta, ...)
    padded_kwargs = model.fill_buffers_cudagraph(self.meta, ...)   # copy inputs
    model.update_context_cudagraph(self.meta, context)             # update ctx

    warmup_output = model(**padded_kwargs)                        # eager warmup

    self._graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(self._graph, pool=self.pool, stream=stream):
        output = model(**padded_kwargs)                            # ← CAPTURED

    self.meta.output_buffers = model.make_output_buffers(output)
    return model.get_outputs_cudagraph(warmup_buffers, ...)

# REPLAY (subsequent occurrences)
def forward(self, **kwargs):
    model.fill_buffers_cudagraph(self.meta, ...)        # overwrite buffer data
    model.update_context_cudagraph(self.meta, context)  # update ctx pointers
    self._graph.replay()                                 # zero CPU overhead
    return model.get_outputs_cudagraph(self.meta.output_buffers, ...)
```

## SSM / State Cache Special Case

SSM/linear-attention models have state caches that are updated in-place during
forward. Current main passes state cache tensors through `StepContext` and pads
state slot ids through CUDAGraph buffers as `state_ids`:

```python
# make_buffers_cudagraph
if graph_meta.is_ssm:
    input_buffers['state_ids'] = torch.full((max_batches,), -1, dtype=torch.int64, ...)

# fill_buffers_cudagraph
if graph_meta.is_ssm:
    state_ids = kwargs['state_ids']
    input_buffers['state_ids'].fill_(-1)
    input_buffers['state_ids'][:state_ids.size(0)].copy_(state_ids)
    kwargs['state_ids'] = input_buffers['state_ids']
```

Because state updates happen inside the graph, the **warmup capture mutates real state**. The comment in `graph_runner.py` explains:

> "SSM would update the state in capture(warmup), replay the graph will leads unexpected state update."

Therefore the warmup output is returned directly; replay is only safe for decode steps where state evolution is deterministic.

## Checklist for Enabling CUDAGraph on a New Model

Before changing `support_cuda_graph()` from `False` to `True`:

- [ ] Search `model(**input_dict)` call chain for `.item()`, `.tolist()`, `.cpu()`, `int(tensor)`
- [ ] Search for `torch.arange(...)` with non-constant length
- [ ] Search for `torch.cat` with variable-length lists
- [ ] Search for Python `if` / `for` that branch on tensor values
- [ ] Verify all intermediate shapes are bounded by `max_batches`, `max_tokens`, or `num_blocks`
- [ ] Run warmup with `CUDA_LAUNCH_BLOCKING=1` to surface hidden syncs
- [ ] Test decode with varying sequence lengths to ensure shapes are stable
- [ ] Confirm `make_buffers_cudagraph` / `fill_buffers_cudagraph` handle all new inputs
