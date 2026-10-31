---
id: hw-2sm-cooperative
title: "Two-SM Cooperative MMA"
type: hardware
architectures: [sm100, sm100a]
tags: [2sm-cooperative, tcgen05, cluster]
confidence: source-reported
related: [hw-tcgen05-mma, hw-tmem, technique-warp-specialization]
sources: [doc-nvidia-tuning-guide, blog-colfax-cutlass, blog-modular-blackwell]
aliases: ["2-SM cooperative", "dual CTA", "2CTA", "cta_group::2"]
---

## 本工作台的语义修正

两 CTA 协同 MMA、按 M 分片及集群同步可作为候选方向。

stage 已去 TPC 错称并加示意声明；仍把两个 SM 的 TMEM 错画成同一 512 列的前后半段；实际每个 SM 自有 512 列。指令 idesc/C/D 占位不可编译。

正确实现按以下契约落实：按每 CTA/SM 独立 TMEM 银行及 M 维 ownership 重绘；使用固定 CUTLASS two-CTA 原子与集群 barrier。具体同步、数值和映射参考见[已核对的技术契约](../../../../references/kernelwiki-semantic-corrections.md)。

以下上游叙述为历史来源材料；上述冲突处以本修正为准。被指出的 CUDA/PTX/Triton 片段不能直接执行或标为数值通过；其历史性能不计作本次实测。

## 保留的上游叙述


## Overview

Blackwell enables two SMs within a TPC to cooperatively execute a single larger MMA, doubling the effective compute tile size to m256×n256×k16.

## How It Works

```
Cooperating two-CTA cluster (illustrative)
├── SM 0: CTA 0 — issues tcgen05.mma with cta_group::2
│   ├── Shared Memory A (rows 0-127)
│   └── TMEM (columns 0-255)
└── SM 1: CTA 1 — cooperates on same MMA
    ├── Shared Memory A (rows 128-255)
    └── TMEM (columns 256-511)
```

## PTX

```ptx
// 2-SM cooperative MMA
tcgen05.mma.cta_group::2.kind::f16
    [tmem_addr], descA, descB, idescC, idescD, ...;
```

## Requirements
1. **Identical shared memory layouts** across both CTAs
2. `shared::cluster` mbarrier signaling between the two CTAs
3. Both CTAs in the same cluster
4. Each CTA contributes half the M-dimension

## Performance Impact

From tcgen05 tutorial progression:
- 1-SM MMA (m128×n256): 80% of cuBLAS → adding 2-SM: **86%** of cuBLAS
- ~7.5% improvement from doubling the MMA tile size

## When to Use
- Large GEMM problems where M ≥ 256
- Compute-bound kernels where peak FLOPS matters
- Combined with persistent scheduling for maximum throughput

## Related
- [tcgen05-mma](tcgen05-mma.md) — Base MMA instruction
- [tmem](tmem.md) — Full TMEM used in 2-SM mode

Local scope note: the diagram and PTX are conceptual. Use the source implementation for exact TMEM ownership, instruction operands, cluster admission and launch conditions; the percentage progression is source-reported for that example, not a guarantee.
