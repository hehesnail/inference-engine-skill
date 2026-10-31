> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[NVIDIA/TensorRT-LLM@f405e10b730e](https://github.com/NVIDIA/TensorRT-LLM/blob/f405e10b730ea953961ecf5719242c581527587a/.claude/skills/perf-workload-profiling/SKILL.md)，2026-10-31 核对。

# Workload Profiling

## Quick Reference

Pick ONE path based on the workload type:

| Workload | Approach | Section |
|----------|----------|---------|
| Training loop | Manual `torch.cuda.synchronize()` + `time.perf_counter()` with warmup | Loop Workloads — Manual Timing |
| Single kernel or op | Write CUDA event benchmark (pre-allocate, warmup, event pairs) | Non-Loop Workloads — CUDA Event Benchmarking |
| Add timeline labels for nsys | Use `@nvtx.annotate` decorator or context manager | NVTX Reference |

## Principles

- **Measure, don't guess.** Every performance claim must trace back to profiler output or structured measurement data. Never invent metrics.
- **Isolate steady-state.** Warmup costs (CUDA context init, cuDNN autotuning, JIT compilation) distort measurements. Always exclude warmup iterations before collecting data.
- **Choose the timing boundary.** CUDA events measure an ordered device-stream interval; work on other streams needs explicit event dependencies. A wall timer with synchronization at the chosen boundaries measures host plus completed device work. Report which interval is intended; neither is automatically end-to-end serving latency.
- **Preserve the intended overlap.** For steady-state throughput, enqueue event pairs and synchronize once after the measured window. Per-iteration synchronization below is a separate diagnostic latency mode and changes overlap; its overhead is workload-dependent, not a universal 10-50us.
- **Pre-allocate everything.** Tensors, events, compiled kernels — all before the timing loop. For CuTe DSL kernels, pre-compile with `cute.compile()`.
- **Minimize profiler interference.** Start with lightweight measurement (manual timing for latency/throughput) and escalate to heavier tools (Kineto, nsys, ncu) only when lighter tools cannot answer the question.

## Loop Workloads — Manual Timing

For training loops and iterative workloads, use manual `torch.cuda.synchronize()` + `time.perf_counter()` timing with warmup to measure per-iteration latency, throughput, and data load time.

### Injection Template

Read the user's training script, understand the dataloader and loop structure, then inject timing code.

```python
import time
import torch

WARMUP = 5
NUM_ITERS = 30
BATCH_SIZE = 128  # global batch size for throughput calculation

iter_times = []
data_times = []

for i, batch in enumerate(dataloader):
    if i >= WARMUP + NUM_ITERS:
        break

    t_data_end = time.perf_counter()

    torch.cuda.synchronize()
    t_start = time.perf_counter()

    # ... existing training loop body ...

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    if i >= WARMUP:
        iter_ms = (t_end - t_start) * 1000
        iter_times.append(iter_ms)
        if i > 0:
            data_times.append((t_data_end - prev_iter_end) * 1000)
        print(f"[{i:04d}]: iter {iter_ms:.2f} ms, fps {BATCH_SIZE / (iter_ms / 1000):.2f}")

    # Exclude printing/bookkeeping from the following dataloader-wait interval.
    prev_iter_end = time.perf_counter()

import statistics
print(f"Average: iter {statistics.mean(iter_times):.2f} ms, "
      f"fps {BATCH_SIZE / (statistics.mean(iter_times) / 1000):.2f}")
```

### Interpreting Results

- **iter (ms)**: Wall-clock time per iteration (compute + communication, excluding data loading)
- **data (ms)**: Observed host wait for the next batch in this instrumented loop. A large ratio is a lead to inspect, not proof of the original overlapping workload's bottleneck.
- **fps**: Compute-window examples/second here; data loading is excluded. For pipeline throughput count completed examples over the full measurement window. Multi-rank global throughput needs the global sample count and the same window ending at the slowest participating rank; one rank's local timer is insufficient for MFU.

### Limitations

Manual timing reports **aggregate** iteration timing — not per-sub-phase breakdown (forward, backward, optimizer). When the user asks **where time is spent within compute**:

1. Add `torch.cuda.synchronize()` + `time.perf_counter()` around each sub-phase for a one-off diagnosis, OR
2. Add NVTX annotations and run with `nsys profile` for timeline visualization.

## Non-Loop Workloads — CUDA Event Benchmarking

For single kernels, one-shot inference, or standalone operations, write CUDA event benchmarking code directly.

### PyTorch: Simple (Mean Only)

```python
import torch

def benchmark(fn, warmup=50, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters  # ms per iteration
```

### PyTorch: Detailed (Per-Iteration Stats)

```python
import torch
import statistics

def benchmark_detailed(fn, warmup=50, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()

    torch.cuda.synchronize()
    times = [starts[i].elapsed_time(ends[i]) for i in range(iters)]

    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
    }
```

### Anti-Patterns

| Anti-Pattern | Problem |
|--------------|---------|
| Per-iteration synchronization in a steady-state throughput experiment | Serializes host/device overlap; measure its actual cost |
| Unsynchronized wall timer around asynchronous launches | Measures submission rather than completion; a synchronized wall timer is valid for an explicitly defined host+device interval |
| Missing warmup | First iterations include JIT, clock ramp-up, context init |
| Allocating tensors inside measurement loop | Allocation overhead pollutes timing |
| Reporting only mean | Hides variance, outliers, bimodal distributions |

The example callables assume repeatable inputs. A mutating stateful callable (KV/SSM state, RNG, in-place buffers) must restore the agreed initial state outside each timed trial, or explicitly measure a defined progressing workload. Warmup/capture must not consume the benchmark input state. Validate outputs before timing and require positive iteration counts.

For additional benchmarking templates (CUDA Graph, CuTe DSL, Triton, Raw CUDA), see [references/benchmarking-patterns.md](references/benchmarking-patterns.md).

## NVTX Reference

NVTX (NVIDIA Tools Extension) adds named annotations to profiler timelines. Use NVTX to label phases (forward, backward, optimizer) for readability in nsys — not for measurement.

```python
import nvtx

# Decorator — annotates every call
@nvtx.annotate("training_step", color="blue")
def training_step():
    ...

# Context manager — annotates a code block
with nvtx.annotate("data_loading", color="green"):
    batch = next(dataloader)
```

- **Do** annotate training phases (forward, backward, optimizer, data loading) for nsys timeline clarity.
- **Do not** annotate for measurement — use CUDA events or manual timing instead.
- **Do not** over-annotate — too many fine-grained ranges add visual clutter and minor overhead.

For NVTX domains, categories, payloads, and legacy API details, see [references/nvtx-api.md](references/nvtx-api.md).

## References

- [references/benchmarking-patterns.md](references/benchmarking-patterns.md) — CUDA Graph, CuTe DSL, Triton, Raw CUDA templates; warmup guidance; GPU hardware properties; reporting format
- [references/nvtx-api.md](references/nvtx-api.md) — Domains, categories, payloads, legacy push/pop API
- [references/pytorch-profiler-api.md](references/pytorch-profiler-api.md) — PyTorch 2.0+ profiler API changes (`device_time` vs deprecated `cuda_time`)
