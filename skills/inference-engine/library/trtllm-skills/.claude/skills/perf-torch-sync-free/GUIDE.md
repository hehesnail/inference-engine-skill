> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[NVIDIA/TensorRT-LLM@f405e10b730e](https://github.com/NVIDIA/TensorRT-LLM/blob/f405e10b730ea953961ecf5719242c581527587a/.claude/skills/perf-torch-sync-free/SKILL.md)，2026-10-31 核对。

# Writing Sync-Free PyTorch Code

Sync-free code means the CPU continuously queues work to the GPU without
waiting for GPU operations to complete. Removing unnecessary host waits can improve overlap; it does not guarantee that kernels, data or dependencies are ready enough to keep the GPU busy.

Common explicit waits include `cuEventSynchronize`, `cuStreamSynchronize` and `cuCtxSynchronize`. Blocking copies, allocation or library/runtime internals can also wait. Inspect the actual API timeline; absence of these three calls does not prove absence of host blocking.

## When to Use

Reach for this skill when you encounter:

- **Triggers**: User wants to remove host-device synchronizations, eliminate
  CPU stalls from GPU waits, make code async/sync-free, remove `.item()` or
  `.cpu()` calls that block the CPU, or understand why specific PyTorch
  operations cause synchronization
- **Symptoms**: Frequent `cudaStreamSynchronize` in nsys profiles,
  warnings from `torch.cuda.set_sync_debug_mode`, training throughput
  limited by CPU-GPU round-trips, `.item()` or `.cpu()` calls in hot loops
- **Keywords**: "sync-free", "synchronization", ".item()", ".cpu()",
  "host-device sync", "eliminate syncs", "CPU stall", "non_blocking",
  "set_sync_debug_mode", "cudaStreamSynchronize", "cudaEventSynchronize",
  "remove syncs", "async GPU", "CPU waiting on GPU"

Do NOT use this skill for:

- Applying CUDA Graphs or reducing kernel launch overhead (use
  `perf-torch-cuda-graphs` instead)
- Profiling GPU kernels, system timelines, or finding GPU idle time (use
  `perf-nsight-compute-analysis` or `perf-nsight-systems`)
- Kernel optimization or code generation (use `kernel-triton-writing`)
- Optimizing NCCL communication or distributed training collective
  operations
- Reducing GPU memory usage or gradient checkpointing
- General model compilation with `torch.compile`

## Requirements

| Dependency | Version | Notes |
|------------|---------|-------|
| PyTorch | >=2.0 | With CUDA support |
| NVIDIA GPU | Any | CUDA-capable |
| Nsight Systems | Optional | For comprehensive sync detection via `nsys` |

## Workflow

### Step 1: Detect Synchronizations

Use one or both methods to find sync points in the code.

**Quick detection** — PyTorch sync debug mode warns/errors for supported synchronization paths; it is experimental and does not cover every operation (including some distributed/sparse and extension paths).

```python
import torch

# Enable at the start of the region you want to check
torch.cuda.set_sync_debug_mode('warn')   # prints warning + stack trace
# torch.cuda.set_sync_debug_mode('error')  # raises exception on sync

# Run your training step / forward pass here
train_step(model, batch)

torch.cuda.set_sync_debug_mode(0)  # disable
```

Treat a quiet debug run as incomplete screening, not a proof. Correlate profiler APIs, CPU waits and dependency traces for the actual workload.

**Broader detection** -- Nsight Systems can capture traced CUDA wait calls
including those from extensions and libraries:

```bash
nsys profile --capture-range=cudaProfilerApi \
             --python-sampling=true \
             --backtrace=dwarf --cudabacktrace=all --python-backtrace=cuda \
             python your_script.py
```

In the Nsight Systems GUI, check the **CUDA API** timeline row and search
for `cudaStreamSynchronize`, `cudaEventSynchronize`, or
`cudaDeviceSynchronize`. When backtrace capture is enabled and supported, the call stack panel can attribute captured waits. With `cudaProfilerApi`, the application must call cudaProfilerStart/Stop around the selected region.

### Step 2: Classify -- False vs True Dependencies

After detecting syncs, classify each one before deciding how to fix it.

**False dependencies** (avoidable) -- CPU does not actually need the GPU
result. These can be eliminated without changing program logic:

- Debug prints left in hot paths (`print(loss.item())`)
- Unnecessary `.item()` calls for logging that could be deferred
- Using `.cuda()` instead of `.to('cuda', non_blocking=True)`
- Using `.type(torch.LongTensor)` instead of `.type(torch.long)`
- Creating tensors from Python objects directly on CUDA

**True dependencies** (require restructuring) -- CPU genuinely needs the
GPU value to proceed:

- **Control flow dependency**: `if loss.item() > threshold:` -- CPU
  branches on a GPU-computed value
- **Dynamic memory allocation**: `output = x[mask]` -- output size depends
  on GPU computation
- **CPU computation using GPU values**: computing statistics for logging,
  updating learning rates from metrics

True dependencies require restructuring: move logic to GPU
(`torch.where()`), delay to end of iteration, or accept that those parts
stay outside any CUDA Graph capture region.

### Step 3: Eliminate Systematically

Apply fixes in order of increasing difficulty. Start with easy wins.

**1. Remove redundancy** -- Delete operations that do not need to happen:
- Remove debug prints and logging from hot loops
- Delete unnecessary `.item()` calls
- Eliminate duplicate synchronizations

**2. Use `non_blocking=True`** -- Make transfers async where CPU does not
immediately use the result:

```python
# H2D: a pinned source can support overlap, but keep it alive and unmodified
# until the transfer completes; manage cross-stream dependencies explicitly.
x_gpu = x_cpu.to('cuda', non_blocking=True)

# D2H: record completion after the copy on its stream.
x_cpu = x_gpu.to('cpu', non_blocking=True)
done = torch.cuda.Event()
done.record(torch.cuda.current_stream(x_gpu.device))
# ... independent CPU work ...
done.synchronize()  # required before any CPU consumer reads x_cpu
use_on_cpu(x_cpu)
```

`non_blocking=True` requests asynchronous behavior where supported; pageability, staging and backend details affect overlap. A delayed CPU read is safe only after a known completion dependency, not merely after some elapsed time.

**3. Switch to sync-free API alternatives** -- See the Quick Reference
Table below for a condensed mapping of common patterns.

**4. Delay synchronization to end of iteration** -- Move logging and
validation to after the optimizer step rather than mid-forward/backward:

```python
# Before: sync mid-iteration
loss = model(batch)
print(f"Loss: {loss.item()}")    # cuStreamSynchronize
loss.backward()

# After: delay to end of iteration
loss = model(batch)
loss.backward()
optimizer.step()
print(f"Loss: {loss.item()}")    # wait remains in per-iteration latency; it is only moved later
```

**5. Coalesce multiple syncs into one** -- If you need several GPU values
on CPU, gather them and transfer once:

```python
# Before: 3 separate syncs
loss_val = loss.item()           # cuStreamSynchronize
acc_val = accuracy.item()        # cuStreamSynchronize
gnorm_val = grad_norm.item()     # cuStreamSynchronize

# After: 1 sync
metrics = torch.stack([loss, accuracy, grad_norm])
vals = metrics.cpu()             # single cuStreamSynchronize
loss_val, acc_val, gnorm_val = vals.tolist()
```

**6. Offload logic to GPU** -- Use selection only when both branches are safe to compute, side-effect-free and shape-compatible; `torch.where` does not short-circuit Python evaluation:

```python
# Before: CPU control flow (syncs)
if loss.item() > threshold:
    result = a
else:
    result = b

# After: GPU-side selection (no sync)
result = torch.where(loss > threshold, a, b)

# Before: Python max (syncs)
val = max(x_gpu[0, 0], x_gpu[0, 1])

# After: torch.max (no sync)
val = torch.max(x_gpu[0, 0], x_gpu[0, 1])
```

**7. Exclude unavoidable syncs from capture range** (last resort) -- If a
sync cannot be eliminated, keep it outside the CUDA Graph capture region
and graph only the sync-free sections. Partial graphing is better than no
graphing.

### Step 4: Verify

Re-run detection to confirm syncs are eliminated:

```python
torch.cuda.set_sync_debug_mode('error')  # raises for covered sync paths only
train_step(model, batch)
torch.cuda.set_sync_debug_mode(0)
```

Or re-profile with Nsight Systems and confirm no `cudaStreamSynchronize` /
`cudaEventSynchronize` / `cudaDeviceSynchronize` calls appear in the
target region.

## Quick Reference Table

| Sync-Inducing Pattern | Sync-Free Alternative |
|----------------------|----------------------|
| **Device Transfers** | |
| `.cpu()` or `.to('cpu')` | `.to('cpu', non_blocking=True)` plus completion event before CPU consumption |
| `.cuda()` or `.to('cuda')` | `.to('cuda', non_blocking=True)` |
| `.type(torch.LongTensor)` | `.type(torch.long)` (dtype conversion, stays on GPU) |
| **Tensor Creation** | |
| `torch.tensor(obj, device='cuda')` | Create on CPU, then `.to('cuda', non_blocking=True)` |
| `torch.tensor(0, device='cuda')` | `torch.zeros(1, device='cuda', dtype=...).squeeze()` |
| `torch.as_tensor(arr, device='cuda')` | Create on CPU, then `.to('cuda', non_blocking=True)` |
| `torch.cuda.BoolTensor(list)` | `torch.tensor(list, device='cpu').to('cuda', non_blocking=True)` |
| **Control Flow** | |
| `.item()` in conditionals | `torch.where()` or move outside critical region |
| `if gpu_tensor:` | Keep logic on GPU with `torch.where()` |
| Python `max(a, b)` on GPU tensors | `torch.max(a, b)` |
| `torch.is_nonzero(t)` | Avoid; use GPU-side comparisons |
| **Indexing** | |
| `x_gpu[idx_cpu]` or `x_gpu[idx_list]` | `x_gpu[idx_gpu]` (keep indices on same device) |
| Indexed assignment | Profile the exact indexing/value path; Python scalar assignment is not universally synchronizing, and a GPU value is not a universal fix |
| `x[i:j]` with CUDA tensor bounds | Use known host bounds, or a fixed-size device index/mask representation if the consumer accepts it; CUDA scalar bounds to arange still need host values |
| **Dynamic Shapes** | |
| `x_gpu[mask_gpu]` (masked selection) | Fixed-shape masked output only if downstream accepts original shape, mask and fill semantics |
| `torch.nonzero(mask)` | One-argument `torch.where(mask)` also has dynamic output; use a verified fixed-size representation or move the operation |
| `torch.masked_select(x, mask)` | A fixed-shape replacement changes output shape/order semantics; adapt and validate its consumer |
| `torch.unique(x)` | Avoid in hot path; precompute if possible |
| `torch.repeat_interleave(x, r)` | Specify `output_size=N` if known |

## Finding More Information

- **Tier 1 (this file)**: Workflow, classification, elimination strategies,
  and quick reference table
- **Tier 2 (`references/sync-patterns.md`)**: Comprehensive pattern catalog
  with 9 categories, full code examples showing sync-inducing and sync-free
  versions, and the specific CUDA driver API triggered by each pattern

Primary contracts for the corrections above (verify installed versions): [official source 1](https://docs.pytorch.org/docs/stable/generated/torch.cuda.set_sync_debug_mode.html), [official source 2](https://docs.pytorch.org/docs/stable/notes/cuda.html).
