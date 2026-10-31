> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[NVIDIA/TensorRT-LLM@f405e10b730e](https://github.com/NVIDIA/TensorRT-LLM/blob/f405e10b730ea953961ecf5719242c581527587a/.claude/skills/perf-nsight-systems/SKILL.md)，2026-10-31 核对。

# Nsight Systems Profiling

NVIDIA Nsight Systems (`nsys`) is a system-level performance analysis tool
that captures CPU/GPU activity timelines, API traces, and OS-level events.
Unlike Nsight Compute (kernel-level), `nsys` shows the big picture — how
kernels, memory transfers, communication, and CPU work overlap in time.

## When to Use

Reach for this skill when you encounter:

- **Triggers**: User wants to profile a training script end-to-end, analyze
  GPU utilization, find pipeline bottlenecks, check communication/compute
  overlap, or interpret `.nsys-rep` reports
- **Symptoms**: Training slower than expected, GPU idle between iterations,
  need to understand where time is spent across CPU and GPU, poor scaling
  in distributed training
- **Keywords**: "nsys", "nsight systems", "GPU timeline", "GPU utilization",
  "kernel launch overhead", "training profiling", "NCCL overlap", "nsys-rep",
  "cuda trace", "GPU idle", "pipeline stall", "data loading bottleneck"

Do NOT use this skill for:
- Kernel-level optimization (use Nsight Compute / `ncu` instead)
- GPU hardware metrics like SM throughput, cache hit rates (use `ncu`)
- GPU monitoring without profiling (use `nvidia-smi`)
- Non-CLI usage (GUI workflows, IDE integration) — consult official docs

## Requirements

| Dependency | Version | Notes |
|------------|---------|-------|
| CUDA Toolkit | >=11.0 | Includes `nsys` |
| `nsys` binary | Match CUDA version | Verify with `nsys -v` |
| NVIDIA GPU | Any supported | |

Permissions: `nsys` may require `sudo` or `CAP_SYS_ADMIN` for system-wide
tracing and GPU metrics. In containers, use `--privileged` or
`--cap-add=SYS_ADMIN`.

## Reporting Principles

**Every number must have an authoritative source.** When presenting timing data,
kernel counts, API call durations, or any quantitative metric, always cite the
source: `nsys stats` report output, `nsys analyze` rule output, exported SQLite
query result, recipe CSV, or raw command output. Show the actual command and its
output before interpreting. Never synthesize, estimate, or extrapolate numbers
that did not come from a tool output.

**Use `nsys stats` for structured analysis, not raw trace data.** Always extract
metrics via targeted `nsys stats -r <report>` commands rather than trying to
read or interpret `.nsys-rep` files directly. Stats reports produce compact,
tabular summaries; raw trace data can be enormous (especially with backtraces
or verbose API tracing). Run the smallest set of reports needed for the task,
then request additional reports only if the initial results raise questions.

## Workflows

### Workflow 1: Profile a DL Training Script

Goal: Capture a clean, focused profile of steady-state training iterations.

**Step 1 — Add profiler markers** to your training script to skip warmup:

```python
assert profile_iters > 0
for i, batch in enumerate(train_loader):
    if i == warmup_iters:
        torch.cuda.synchronize()  # drain warmup before starting collection
        torch.cuda.cudart().cudaProfilerStart()
    train_step(model, batch)
    if i == warmup_iters + profile_iters - 1:
        torch.cuda.synchronize()  # complete the final measured iteration
        torch.cuda.cudart().cudaProfilerStop()
        break
```

**Step 2 — Profile with `cudaProfilerApi` capture range:**

```bash
nsys profile -c cudaProfilerApi \
    -t cuda,nvtx,cudnn,cublas \
    --pytorch=autograd-nvtx \
    -o train_profile -- python train.py
```

This captures only steady-state iterations — no warmup, no initialization noise.

Note: `-t cuda,nvtx,cudnn,cublas` enables **API-specific tracing**. By default,
`-t cuda` only traces the CUDA runtime/driver layer — you see kernel names and
launch times but cannot attribute them to higher-level libraries. Adding `cudnn`
and `cublas` traces the library-level API calls, letting you distinguish
convolution time (cuDNN) from GEMM time (cuBLAS) and measure library overhead
separately from raw kernel execution.

**Step 3 — Quick summary:**

```bash
nsys stats -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum \
    train_profile.nsys-rep
```

When you traced library APIs (`cudnn`, `cublas` in `-t`), also run the
library-specific reports to see API-level overhead (workspace allocation,
algorithm selection) separately from raw kernel execution:

```bash
nsys stats -r cudnn_api_sum,cublas_api_sum train_profile.nsys-rep
```

**Step 4 — Detect anti-patterns:**

```bash
nsys analyze -r all train_profile.nsys-rep
```

**Step 5 — Dig deeper** based on findings. See Tier 2 references.

### Workflow 2: Diagnose GPU Idle Time

Goal: Find why the GPU is idle between training iterations.

**Step 1 — Profile with OS runtime tracing:**

```bash
nsys profile -t cuda,nvtx,osrt \
    --pytorch=autograd-nvtx \
    -o idle_debug -- python train.py
```

**Step 2 — Check GPU gaps and utilization:**

```bash
nsys analyze -r gpu_gaps,gpu_time_util idle_debug.nsys-rep
```

**Step 3 — Check kernel launch phases:**

```bash
nsys stats -r cuda_kern_exec_sum idle_debug.nsys-rep
```

Queue delay can include outstanding work, events and other dependencies; a short queue can also reflect intentionally serialized work. Correlate stream/event/CPU readiness and actual GPU activity before concluding that the host starved the GPU.

**Step 4 — Common causes and fixes:**

| GPU idle cause | Evidence | Fix |
|---------------|----------|-----|
| Slow data loading | CPU busy in DataLoader during gaps | Increase `num_workers`, use `pin_memory=True` |
| Synchronous memcpy | `cuda_memcpy_sync` rule fires | Use `non_blocking=True` transfers |
| Over-synchronization | Frequent `cudaDeviceSynchronize` in trace | Remove unnecessary sync calls |
| Host-side computation | CPU sampling shows compute during gaps | Move to GPU or overlap with async ops |
| Python GIL contention | GIL trace shows contention | Use multiprocessing, reduce Python overhead |

### Workflow 3: Profile Distributed Training

Goal: Profile multi-GPU/multi-node training with communication analysis.

**Step 1 — Collect per-rank profiles:**

```bash
# Single node: torchrun launches one profiler per worker, where RANK is defined.
torchrun --standalone --nnodes=1 --nproc-per-node=<gpus> --no-python \
    nsys profile -t cuda,nvtx -o 'profile_%q{RANK}' -- python train.py
```
For multiple nodes, launch one torchrun per node. Use c10d rendezvous with shared nnodes/endpoint/id (worker ranks are assigned), or static rendezvous with explicit node rank and master address/port. The node-rank option is used by static rendezvous, not c10d. Confirm profiler options and output paths are supported on every node.

**Step 2 — Analyze NCCL communication/compute overlap:**

```bash
nsys recipe nccl_gpu_overlap_trace -- profile_*.nsys-rep
nsys recipe nccl_gpu_time_util_map -- profile_*.nsys-rep
```

**Step 3 — Check per-rank utilization:**

```bash
nsys recipe cuda_gpu_time_util_map -- profile_*.nsys-rep
```

**Step 4 — Check for stragglers:**

Compare `cuda_gpu_kern_sum` across ranks. If one rank is slower, check
its network and data loading patterns.

### Workflow 4: Analyze Iteration Time Consistency

Goal: Check whether training iterations are stable or have outliers.

```bash
# Profile with NVTX iteration markers
nsys profile --pytorch=autograd-nvtx -t cuda,nvtx \
    -o iter_check -- python train.py

# Check iteration timing distribution
nsys stats -r nvtx_pushpop_sum iter_check.nsys-rep

# Check GPU projection per NVTX range
nsys stats -r nvtx_gpu_proj_sum iter_check.nsys-rep

# Visual pace analysis
nsys recipe nvtx_pace -- iter_check.nsys-rep
```

High StdDev in iteration duration indicates inconsistency — investigate
outlier iterations on the timeline.

### Workflow 5: Attribute Kernels to Source Code via Stack Traces

Goal: Identify which Python function or code path triggers expensive GPU kernels.

**Step 1 — Profile with backtrace collection:**

```bash
nsys profile -t cuda,nvtx \
    --cudabacktrace=all --backtrace=dwarf \
    --python-backtrace=cuda \
    --pytorch=autograd-nvtx \
    -o stacktrace_profile -- python train.py
```

- `--cudabacktrace=all` selects CUDA API backtraces; `--backtrace=dwarf` selects the CPU unwinding method. `--python-backtrace=cuda` requests Python frames for traced CUDA APIs. Check the installed `nsys profile --help` and sampling/platform requirements; LBR is a CPU unwinding method, not a Python-backtrace API selector.

**Step 2 — Get kernel summary and NVTX attribution:**

Use targeted stats reports to identify top kernels and their NVTX context:

```bash
# Top kernels by total GPU time
nsys stats -r cuda_gpu_kern_sum stacktrace_profile.nsys-rep

# Kernels attributed to NVTX ranges (maps kernels to annotated code regions)
nsys stats -r nvtx_kern_sum stacktrace_profile.nsys-rep
```

The `nvtx_kern_sum` report (requires `--pytorch=autograd-nvtx` or manual NVTX
annotations) maps each kernel to its enclosing NVTX range, directly showing
which Python function or autograd op launched it. This is more efficient than
manually cross-referencing raw backtrace data.

**Step 3 — For PyTorch models**, `--pytorch=autograd-nvtx` automatically wraps
each autograd op in an NVTX range. Combined with backtrace, this maps:
GPU kernel → CUDA API call → Python function → PyTorch autograd op.

**When to use**: Workloads with multiple code paths launching similar kernels
(e.g., attention vs FFN both calling GEMM). Stack traces disambiguate which
caller is responsible for the dominant kernel time.

## Output Formats

**Report files** (`.nsys-rep`): Binary format, viewable in GUI or processed
with `nsys stats`, `nsys analyze`, `nsys export`, `nsys recipe`.

**Stats output formats**: `column` (terminal), `csv`, `json`, `table`, `tsv`,
`hdoc`, `htable`.

**Export formats**: `sqlite` (SQL queries), `arrow`/`parquetdir` (Pandas/Dask),
`hdf`, `jsonlines`, `text`.

**Recipe output**: Directory with CSV/Parquet data + Plotly HTML visualizations
+ `.nsys-analysis` (Jupyter notebook).

**Key stats report columns:**

| Report | Key columns |
|--------|------------|
| `cuda_gpu_kern_sum` | Time%, Total Time, Instances, Kernel Name |
| `cuda_api_sum` | Time%, Total Time, Num Calls, API Name |
| `cuda_kern_exec_sum` | API Time, Queue Time, Kernel Time |
| `cuda_gpu_mem_time_sum` | Time%, Total Time, Operations, Direction |
| `nvtx_gpu_proj_sum` | Projected Duration, Original Duration, GPU Op Count |

## Examples

### Example 1: Quick DL Profile and Summary

```bash
# Profile
nsys profile -t cuda,nvtx,cudnn,cublas \
    --pytorch=autograd-nvtx --stats=true \
    -o quick_profile -- python train.py

# Auto-generates stats at the end of profiling
```

### Example 2: Detect Sync Memcpy in DataLoader

```bash
nsys profile -t cuda,nvtx -o dataloader_check -- python train.py
nsys analyze -r cuda_memcpy_sync,cuda_memcpy_async dataloader_check.nsys-rep
```

If flagged, fix with:
```python
loader = DataLoader(dataset, pin_memory=True, num_workers=4)
tensor_gpu = tensor_cpu.to(device, non_blocking=True)
```

### Example 3: Multi-Node NCCL Analysis

```bash
# Collect one report per worker (single-node example).
torchrun --standalone --nnodes=1 --nproc-per-node=<gpus> --no-python \
    nsys profile -t cuda,nvtx -o 'rank_%q{RANK}' -- python train.py

# Analyze overlap
nsys recipe nccl_gpu_overlap_trace -- rank_*.nsys-rep

# Visualize
nsys recipe nccl_gpu_time_util_map -- rank_*.nsys-rep
```

### Example 4: API-Level Breakdown (cuDNN vs cuBLAS)

```bash
# Profile with library-level tracing
nsys profile -t cuda,nvtx,cudnn,cublas \
    -o api_breakdown -- python model.py

# cuDNN API summary (convolution calls)
nsys stats -r cudnn_api_sum api_breakdown.nsys-rep

# cuBLAS API summary (GEMM calls)
nsys stats -r cublas_api_sum api_breakdown.nsys-rep

# Compare with kernel-level view
nsys stats -r cuda_gpu_kern_sum api_breakdown.nsys-rep
```

Library API summaries measure CPU-side API intervals; kernel summaries measure GPU execution intervals and may overlap each other and CPU work. Subtracting their totals does not measure library overhead. Use correlation IDs and the dependency timeline within the same scope to identify exposed overhead.

## Error Handling

| Error | Cause | Fix |
|-------|-------|-----|
| `nsys: command not found` | Not in PATH | `export PATH=$PATH:/usr/local/cuda/bin` |
| A collector lacks permission | Specific CPU/system metrics may be restricted | Identify the collector and use permitted tracing; request an environment change only if that collector is necessary |
| No CUDA activity captured | App didn't use GPU during collection window | Adjust `--delay`/`--duration`, or use `cudaProfilerApi` capture range |
| Report file very large | Long profile with many APIs traced | Use focused capture (`-c cudaProfilerApi`), reduce `--duration` |
| `--pytorch` has no effect | Wrong nsys version or Python env | Verify nsys version supports `--pytorch`; check Python is in PATH |
| `nsys stats` shows empty reports | No matching activity in report | Check `--trace` flags included the right APIs |
| MPI rank profiles out of sync | Clock skew between nodes | Use NTP sync; analyze per-rank independently |
| `cudaProfilerStart` not captured | Missing `-c cudaProfilerApi` flag | Add `--capture-range=cudaProfilerApi` |
| Recipe fails with import error | Missing Python dependencies | Install recipe dependencies: `pip install pandas plotly` |

## Finding More Information

### Tier 1: This File (SKILL.md)

You are reading it now. The workflows and error table above cover the most
common DL profiling tasks. Search this file first.

### Tier 2: references/ Directory

Grep for keywords across `references/` — headers are grep-friendly:

- `references/cli-profiling.md` — Complete `nsys profile` flags for DL
- `references/cli-post-collection.md` — `nsys stats`, `analyze`, `export`, `recipe` commands
- `references/app-preparation.md` — Focused profiling, NVTX markers, PyTorch patterns
- `references/stats-reports.md` — CUDA statistical report columns and meanings
- `references/expert-systems.md` — Expert system rules, anti-pattern detection
- `references/recipes-dl.md` — DL-relevant advanced recipes with examples
- `references/nvtx-analysis.md` — NVTX statistical reports for annotated code

**How to search:**
1. `Grep` for your keyword across `references/`
2. `Read` only the file that Grep points to

### Tier 3: Official Documentation

If Tiers 1-2 don't answer:
- [User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) — Full CLI reference, all tracing options
- [Analysis Guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html) — Stats reports, expert systems, recipes

WebFetch or WebSearch these URLs for the latest content. Consider distilling
new findings back into `references/`. A distilled file opens with a `**Source:**` paragraph above its first `##` section: a markdown link to the original, `Distilled <YYYY-MM-DD>` and the revision distilled, and what the original still answers.

Primary contracts for the corrections above (verify installed versions): [official source 1](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), [official source 2](https://github.com/pytorch/pytorch/blob/main/torch/distributed/run.py).
