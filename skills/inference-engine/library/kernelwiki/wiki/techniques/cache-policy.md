---
id: technique-cache-policy
title: "PTX Cache Policy Differentiation"
type: technique
architectures: [sm100, sm90]
tags: [cache-policy, vectorized-loads]
confidence: source-reported
reproducibility: snippet
prerequisites: []
related: [technique-vectorized-loads, kernel-nvfp4-gemv, pattern-memory-bound]
sources: [blog-yue-nvfp4, blog-amandeep-nvfp4, blog-simon-nvfp4-gemv, doc-ptx-isa-sm100]
blackwell_relevance: "Same PTX cache hints on both archs; higher B200 bandwidth (8TB/s) amplifies the impact of correct cache policy selection."
---

## 本工作台的语义修正

区分streaming与reuse数据，控制cache pollution可作为候选方向。

cache hint非保证；宽向量指令支持需固定架构；16x多项改动混合不能归因一个cache hint。

正确实现按以下契约落实：记录实际traffic/hitrate、计时边界与配对对照；逐条核对PTX target。具体同步、数值和映射参考见[已核对的技术契约](../../../../references/kernelwiki-semantic-corrections.md)。

以下上游叙述为历史来源材料；上述冲突处以本修正为准。被指出的 CUDA/PTX/Triton 片段不能直接执行或标为数值通过；其历史性能不计作本次实测。

## 保留的上游叙述


# Cache Policy Differentiation

## Overview

PTX cache qualifiers (`L1::no_allocate`, `L1::evict_last`, `L1::evict_first`) let kernels hint to hardware how to handle cache admission for specific loads. Critical for memory-bound kernels where the L1 working set matters more than the compute.

## Pattern

```asm
; Matrix A (streamed once per row, never reused): bypass L1
; Avoids polluting L1 with one-shot data
ld.global.L1::no_allocate.v4.u64 {a0,a1,a2,a3}, [addr_a];

; Vector B (reused across BLOCK_M rows): keep in L1
ld.global.L1::evict_last.v4.u64 {b0,b1,b2,b3}, [addr_b];

; Streaming output: evict immediately after write
st.global.L1::evict_first.v2.u64 [addr_c], {c0, c1};
```

## GPU Mode NVFP4 GEMV Winner Technique

Rank 1 submission used **different qualifiers per K-dimension variant**:
- K=16384 (large): aggressive `L1::no_allocate` on A (huge streaming matrix)
- K=2048 (small): relaxed balance since B is smaller relative to cache

## Measurable Impact

- NVFP4 GEMV: 443μs → 27μs (16x improvement) came partly from cache policy + PTX byte unpacking
- On memory-bound kernels, cache policy can be the dominant lever

## When To Use

- Memory-bound kernels (profile with Nsight Compute first)
- Tensor with clear "streaming" vs "reused" access patterns
- Inputs > L2 cache size (B200: 126MB)
- Separate M and N tile loading patterns in GEMM
