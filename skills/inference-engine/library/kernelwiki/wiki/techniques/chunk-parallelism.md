---
id: technique-chunk-parallelism
title: "Chunk-Based Parallelism for Linear Attention"
type: technique
architectures: [sm100, sm90]
tags: [chunk-parallelism, linear-attention, gated-delta-net, pipeline-stages]
confidence: source-reported
reproducibility: snippet
prerequisites: []
related: [kernel-gated-delta-net, kernel-nsa]
sources: [blog-gated-delta-net, blog-nsa, doc-tfla]
blackwell_relevance: "Chunk size scales with TMEM capacity on Blackwell; larger chunks = better tensor core utilization."
---

## 本工作台的语义修正

将循环改为chunk并验证recurrent等价可作为候选方向。

多个chunk CTA直接读写同一State有竞态；无prefix-state阶段、因果mask，输出地址遗漏chunk。

正确实现按以下契约落实：分开块内变换/跨chunk状态前缀/输出，给顺序oracle验证初始与最终state；不运行现有伪码。具体同步、数值和映射参考见[已核对的技术契约](../../../../references/kernelwiki-semantic-corrections.md)。

以下上游叙述为历史来源材料；上述冲突处以本修正为准。被指出的 CUDA/PTX/Triton 片段不能直接执行或标为数值通过；其历史性能不计作本次实测。

## 保留的上游叙述


# Chunk-Based Parallelism

## Overview

Linear attention variants (GatedDeltaNet, RetNet, Mamba) have O(n) complexity but naive implementations are sequential. Chunk-based parallelism divides the sequence into chunks of size C, computes within each chunk in parallel (matmul-friendly), and propagates state between chunks sequentially.

## Pattern

```python
@triton.jit
def chunk_parallel_linear_attn(Q, K, V, State, Output,
                                chunk_size: tl.constexpr,
                                d: tl.constexpr):
    # Grid: (num_chunks, num_heads, batch)
    chunk_id = tl.program_id(0)

    # Load chunk of Q, K, V
    q = tl.load(Q + chunk_id * chunk_size * d + offsets)  # [C, d]
    k = tl.load(K + chunk_id * chunk_size * d + offsets)
    v = tl.load(V + chunk_id * chunk_size * d + offsets)

    # Intra-chunk: parallel O(C^2) attention-like compute
    scores = tl.dot(q, tl.trans(k))
    o_intra = tl.dot(scores, v)

    # Inter-chunk: sequential state propagation
    state = tl.load(State)  # From previous chunk
    o_inter = tl.dot(q, state)

    # Combine and update state
    output = o_intra + o_inter
    state = update_state(state, k, v)

    tl.store(Output + offsets, output)
    tl.store(State, state)
```

## Size Tradeoff

- **Small chunks (C=32)**: low latency decode, fewer intermediate materializations
- **Large chunks (C=256-512)**: better tensor core utilization, higher throughput for prefill
- **TFLA (Tiled FLA)**: two-level tiling allows arbitrary chunk sizes via recursive tiling

## When To Use

- Linear attention (O(n) complexity)
- Recurrent state models (Mamba, GatedDeltaNet, Delta Rule)
- Hybrid architectures mixing linear + full attention (Qwen3-Next uses 3:1 ratio)
