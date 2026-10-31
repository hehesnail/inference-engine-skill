---
id: technique-ping-pong-scheduling
title: Ping-Pong Scheduling
type: technique
architectures:
- sm100
tags:
- ping-pong-scheduling
- warp-specialization
- tmem
- pipeline-stages
confidence: source-reported
reproducibility: snippet
prerequisites:
- hw-tmem
- technique-warp-specialization
related:
- kernel-flash-attention-4
- technique-double-buffering
sources:
- blog-flash-attention-4
- doc-flash-attention-4
- blog-tcgen05-tutorial
artifact_dir: artifacts/kernels/ping-pong-scheduling
---

## 本工作台的语义修正

两组工作重叠MMA与softmax/epilogue可作为候选方向。

缺init/completion/单issuer约束，异步读写存在风险；Hopper已可ping-pong如FA3；不是100%吞吐保证。

正确实现按以下契约落实：明确每缓冲区full/free、barrier phase和资源预算；引用实际双consumer pipeline。具体同步、数值和映射参考见[已核对的技术契约](../../../../references/kernelwiki-semantic-corrections.md)。

以下上游叙述为历史来源材料；上述冲突处以本修正为准。被指出的 CUDA/PTX/Triton 片段不能直接执行或标为数值通过；其历史性能不计作本次实测。

## 保留的上游叙述


# Ping-Pong Scheduling

## Overview

Ping-pong scheduling alternates two query tiles within a single CTA so the softmax warpgroup never stalls waiting for MMA. Introduced in FlashAttention-4 to exploit Blackwell's asymmetric hardware (2× tensor cores, same SFU count as Hopper).

## Pattern

```cuda
// Two 128-token query tiles per CTA, alternating through the mainloop
// Warpgroup 0: softmax for tile A while MMA runs on tile B
// Warpgroup 1: softmax for tile B while MMA runs on tile A

__global__ void fa4_ping_pong_attn(...) {
    int wg = warp_group_id();

    // TMEM holds accumulators for BOTH tiles
    uint32_t tmem_A = tmem_alloc(256);
    uint32_t tmem_B = tmem_alloc(256);

    for (int k = 0; k < num_kv_tiles; k++) {
        if (wg == 0) {
            // Compute softmax for tile A (previous MMA output)
            softmax_normalize(tmem_A);
            // Issue MMA for tile B next
            tcgen05_mma(Q_B_smem, K_smem[k], tmem_B);
        } else {
            softmax_normalize(tmem_B);
            tcgen05_mma(Q_A_smem, K_smem[k], tmem_A);
        }
        mbarrier_arrive(&ping_pong_sync);
        mbarrier_wait(&ping_pong_sync);
    }
}
```

## Why It Helps on Blackwell

- Tensor core throughput doubled (B200 vs H100) but SFU count unchanged
- Single-tile schedule would leave SFU idle while MMA runs, and vice versa
- Ping-pong keeps both units 100% busy
- FA4 achieves 1605 TFLOPS BF16 (71% utilization) with this pattern

## When To Use

- Compute-bound attention kernels on Blackwell
- Kernels where softmax/epilogue is SFU-heavy
- Not useful on Hopper (balance is different)

## Full Reference Implementation

Verbatim upstream code lives in [`artifacts/kernels/ping-pong-scheduling/full/`](https://github.com/mit-han-lab/KernelWiki/tree/76d27b56f804e7e7295d4c570e1e5d7eef4b0a75/artifacts/kernels/ping-pong-scheduling/full); labeled derived variants (each with the required `// provenance: derived from ...; not upstream code` header) live in [`artifacts/kernels/ping-pong-scheduling/variants/`](https://github.com/mit-han-lab/KernelWiki/tree/76d27b56f804e7e7295d4c570e1e5d7eef4b0a75/artifacts/kernels/ping-pong-scheduling/variants). Every file's SHA-256 and upstream-pinning metadata is in `PROVENANCE.yaml` inside each bundle.

Query via:

```bash
python3 scripts/get_page.py technique-ping-pong-scheduling --include-code
```
