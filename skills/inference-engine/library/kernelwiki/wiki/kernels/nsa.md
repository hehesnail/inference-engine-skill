---
id: kernel-nsa
title: "Native Sparse Attention (NSA)"
type: kernel
architectures: [sm90, sm100]
tags: [sparse-attention, attention, triton]
confidence: source-reported
reproducibility: snippet
kernel_types: [sparse-attention, attention]
languages: [triton]
related: [kernel-flashmla, technique-pipeline-stages]
sources: [blog-nsa, blog-flashmla, blog-vllm-deepseek-v3-sparse]
performance_claims:
  - gpu: H100
    dtype: bf16
    shape: "seqlen=65536"
    metric: speedup
    value: 9.0
    utilization: "vs FlashAttention-2 forward"
    source_id: blog-nsa
blackwell_relevance: "Sparse attention patterns transfer directly; Blackwell's larger L2 (126MB) and higher bandwidth benefit sparse block fetches."
---

## 本工作台的语义修正

块稀疏/滑窗/压缩分支及稳定归一化可作为候选方向。

伪实现用 logaddexp(lse,block_max) 代替 block logsumexp；两个零score时分母只计1，输出翻倍；KV batch/head地址相乘而非线性展开，query offset缺失，跨块stride漏元素；NSA不等同DSA。

正确实现按以下契约落实：给稳定 online softmax 正确不变量、token mask和索引 ABI；区分训练结构 NSA 与 V3.2 DSA。具体同步、数值和映射参考见[已核对的技术契约](../../../../references/kernelwiki-semantic-corrections.md)。

以下上游叙述为历史来源材料；上述冲突处以本修正为准。被指出的 CUDA/PTX/Triton 片段不能直接执行或标为数值通过；其历史性能不计作本次实测。

## 保留的上游叙述


# Native Sparse Attention (NSA)

## Overview

Native Sparse Attention (NSA) is a hardware-aligned sparse attention mechanism published at ACL 2025, designed to reduce attention compute for long sequences (64K+) without sacrificing quality. Unlike post-hoc sparsity approaches, NSA is natively trainable end-to-end. It decomposes attention into three parallel paths -- token compression, token selection, and sliding window -- then fuses their outputs.

NSA achieves 9x forward speedup and 6x backward speedup at 64K sequences versus FlashAttention-2, and 11.6x decoding speedup at 64K context. It is deployed in DeepSeek-V3.2-Exp combined with FlashMLA sparse kernels.

## Three-Path Architecture

```
Input Query Q
    |
    +---> [Compression Path]  Learned MLP creates coarse-grained KV
    |         |                representations (token compression)
    |         v
    |     S_compressed = Q @ K_compressed^T
    |
    +---> [Selection Path]    Blockwise importance scores select
    |         |                top-n fine-grained token blocks
    |         v
    |     S_selected = Q @ K_selected^T   (sparse)
    |
    +---> [Sliding Window]    Local context window (w=512)
              |
              v
          S_local = Q @ K_local^T         (banded)

Output = Combine(softmax(S_compressed) @ V_compressed,
                 softmax(S_selected) @ V_selected,
                 softmax(S_local) @ V_local)
```

## Triton Kernel Implementation

The core NSA kernel is implemented in Triton with group-centric data loading and grid-based scheduling.

```python
import triton
import triton.language as tl

@triton.jit
def nsa_sparse_attention_fwd(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    block_indices_ptr,  # Selected block indices per query
    num_selected: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,  # GQA group size
):
    """
    Sparse attention forward: only compute attention over selected KV blocks.
    Group-centric loading: shares sparse KV blocks across all query heads
    in a GQA group, minimizing redundant KV transfers.
    """
    pid = tl.program_id(0)
    head_id = tl.program_id(1)
    batch_id = tl.program_id(2)

    # GQA: determine which KV head group this query head belongs to
    kv_head_id = head_id // GROUP_SIZE

    # Load query tile
    q_offset = batch_id * NUM_HEADS * HEAD_DIM + head_id * HEAD_DIM
    q = tl.load(Q_ptr + q_offset + tl.arange(0, HEAD_DIM))

    # Accumulator for online softmax
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    lse = float("-inf")

    # Iterate over selected blocks (sparse)
    for block_idx in range(num_selected):
        # Load block index for this query
        sel_offset = (batch_id * NUM_HEADS + kv_head_id) * num_selected + block_idx
        kv_block_start = tl.load(block_indices_ptr + sel_offset) * BLOCK_SIZE

        # Group-centric: load KV block once, shared across GROUP_SIZE query heads
        k_offsets = kv_block_start + tl.arange(0, BLOCK_SIZE)
        k_block = tl.load(K_ptr + (batch_id * kv_head_id * SEQ_LEN + k_offsets[:, None]) * HEAD_DIM
                          + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(V_ptr + (batch_id * kv_head_id * SEQ_LEN + k_offsets[:, None]) * HEAD_DIM
                          + tl.arange(0, HEAD_DIM)[None, :])

        # Compute attention scores for this block
        scores = tl.sum(q[None, :] * k_block, axis=1)  # [BLOCK_SIZE]

        # Online softmax update
        block_max = tl.max(scores)
        new_lse = tl.where(lse > block_max,
                           lse + tl.log(1.0 + tl.exp(block_max - lse)),
                           block_max + tl.log(1.0 + tl.exp(lse - block_max)))

        # Rescale accumulator and add new contribution
        old_scale = tl.exp(lse - new_lse)
        new_scale = tl.exp(scores - new_lse)
        acc = acc * old_scale + tl.sum(new_scale[:, None] * v_block, axis=0)
        lse = new_lse

    # Store output
    o_offset = batch_id * NUM_HEADS * HEAD_DIM + head_id * HEAD_DIM
    tl.store(O_ptr + o_offset + tl.arange(0, HEAD_DIM), acc)
```

## Sliding Window Component

```python
@triton.jit
def nsa_sliding_window_fwd(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    seq_pos,
    WINDOW_SIZE: tl.constexpr,  # 512
    HEAD_DIM: tl.constexpr,
):
    """Local sliding window attention for recent context."""
    pid = tl.program_id(0)  # query position

    # Window bounds
    window_start = tl.maximum(0, seq_pos - WINDOW_SIZE)
    window_end = seq_pos

    # Standard dense attention within window
    # This is O(w*d) per query, where w=512
    q = tl.load(Q_ptr + pid * HEAD_DIM + tl.arange(0, HEAD_DIM))
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    lse = float("-inf")

    for pos in range(window_start, window_end, BLOCK_SIZE):
        k = tl.load(K_ptr + pos * HEAD_DIM + tl.arange(0, HEAD_DIM))
        v = tl.load(V_ptr + pos * HEAD_DIM + tl.arange(0, HEAD_DIM))
        score = tl.sum(q * k)
        # Online softmax accumulation
        new_lse = tl.where(lse > score,
                           lse + tl.log(1 + tl.exp(score - lse)),
                           score + tl.log(1 + tl.exp(lse - score)))
        acc = acc * tl.exp(lse - new_lse) + v * tl.exp(score - new_lse)
        lse = new_lse

    tl.store(O_ptr + pid * HEAD_DIM + tl.arange(0, HEAD_DIM), acc)
```

## Hardware-Aligned Design

NSA's sparsity pattern is explicitly designed for GPU memory access efficiency:

1. **Blockwise memory access**: Selected tokens are organized in contiguous blocks (not scattered individual tokens), exploiting spatial locality for contiguous GPU memory loads
2. **Group-centric loading**: In GQA configurations, all query heads in a group share the same sparse KV blocks, minimizing redundant KV transfers to shared memory
3. **Grid-based scheduling**: Triton grid dimensions map directly to (query_block, head, batch), avoiding dynamic scheduling overhead

## Performance

| Sequence Length | Forward Speedup | Backward Speedup | Decoding Speedup |
|----------------|-----------------|-------------------|-------------------|
| 64K | 9.0x | 6.0x | 11.6x |

All speedups measured against FlashAttention-2 on H100 with BF16 precision.

## When to Use

- Long-context inference (32K+ tokens) where full attention is prohibitively expensive
- Models with GQA (grouped query attention) where KV sharing amplifies sparse access efficiency
- End-to-end trainable settings requiring differentiable sparsity

## Caveats

- The compression path requires a learned MLP, adding parameters and training cost
- Token selection adds a two-pass overhead (score all blocks, then select top-n)
- Triton implementation has CPU launch overhead impacting small-batch decode; CUDA graph mode recommended
- Quality depends on training the sparsity selection jointly with the model

## Sources

- [NSA paper (ACL 2025)](https://arxiv.org/abs/2502.11089)
- [PyTorch reference implementation](https://github.com/lucidrains/native-sparse-attention-pytorch)
