---
id: technique-register-budgeting
title: "Register Budgeting for Occupancy"
type: technique
architectures: [sm100, sm90]
tags: [register-budgeting, register-reuse]
confidence: source-reported
reproducibility: snippet
prerequisites: []
related: [pattern-memory-bound, pattern-register-pressure, kernel-nvfp4-gemv]
sources: [blog-yue-nvfp4, blog-amandeep-nvfp4, blog-simon-nvfp4-gemv]
blackwell_relevance: "TMEM eliminates accumulator register pressure on Blackwell, freeing ~100 registers/thread for other uses; technique still critical for memory-bound kernels."
---

## 本工作台的语义修正

减少寄存器的潜在occupancy收益与spill代价可作为候选方向。

stage已有范围限制；正文反比occupancy仍忽略CTA/SMEM/TMEM限制；竞赛不同名次maxreg值不是因果证据。

正确实现按以下契约落实：保留stage限制并改为受多资源限制；相同源码工作负载下比较spill/occupancy/time。具体同步、数值和映射参考见[已核对的技术契约](../../../../references/kernelwiki-semantic-corrections.md)。

以下上游叙述为历史来源材料；上述冲突处以本修正为准。被指出的 CUDA/PTX/Triton 片段不能直接执行或标为数值通过；其历史性能不计作本次实测。

## 保留的上游叙述


# Register Budgeting

## Overview

SM occupancy is inversely proportional to registers-per-thread. For memory-bound kernels, higher occupancy = more warps to hide memory latency. `-maxrregcount` and `__launch_bounds__` force the compiler to stay within a budget.

## Pattern

```cuda
// Aggressive: 32 registers/thread → ~4 blocks per SM at 256 threads/block
__launch_bounds__(256, 4)
__global__ void gemv_memory_bound(...) {
    // Compiler will spill to local memory if needed
}

// Or via nvcc flag:
// nvcc -maxrregcount=32 -arch=sm_100a ...
```

## Compiler Tradeoffs

Lower register count → compiler may:
- Spill frequently-used values to local memory (bad)
- Recompute values instead of storing them (neutral)
- Use fewer unrolled iterations (bad for compute-bound)

For memory-bound kernels, spills can be hidden by memory latency anyway, so aggressive budgeting often wins.

## GPU Mode NVFP4 GEMV Results

| Rank | Register count | Latency |
|------|---------------|---------|
| 1 | 32 | 18.5μs |
| 3 | 45 | ~20μs |

The measurable difference between 32 and 45 registers shows occupancy dominates for memory-bound NVFP4 GEMV.

## When To Use

- Memory-bound kernels (first priority: occupancy)
- Kernels where register pressure comes from inner loop, not accumulators (TMEM handles accumulators)
- Sub-byte types with heavy decode/scale computation

## When NOT To Use

- Compute-bound GEMM (let compiler use what it needs)
- Kernels where spills to local memory would serialize

Local scope note: occupancy, stage count and quoted speedups are hypotheses for this source workload. Check spills, bandwidth/latency, shared-memory per-block versus per-SM limits and the complete pipeline; higher occupancy or deeper staging is not automatically faster.
