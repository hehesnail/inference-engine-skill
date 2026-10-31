---
id: lang-ptx
title: "PTX Instructions for SM100"
type: language
tags: [ptx, tcgen05, tmem, tma, clc, mbarrier, nvfp4]
related: [hw-tcgen05-mma, hw-tmem, hw-clc, lang-cuda-cpp]
sources: [doc-ptx-isa-sm100, doc-nvidia-tuning-guide, blog-yue-nvfp4]
reproducibility: snippet
architectures: [sm100, sm100a]
confidence: source-reported
---

## 本工作台的语义修正

将 ISA 支持和算法设计分别检查可作为候选方向。

tcgen05 alloc/cp/try_cancel/多播限定符有伪语法；fence与completion不可互代；错误示例不能通过有fence就标verified。

正确实现按以下契约落实：固定PTX版本/SM特性并逐条查ISA；对示意片段降级，真实编译/运行证据分开。具体同步、数值和映射参考见[已核对的技术契约](../../../../references/kernelwiki-semantic-corrections.md)。

以下上游叙述为历史来源材料；上述冲突处以本修正为准。被指出的 CUDA/PTX/Triton 片段不能直接执行或标为数值通过；其历史性能不计作本次实测。

## 保留的上游叙述


## Overview

SM100 PTX instructions for Blackwell-specific hardware features.

## tcgen05 Instructions

```ptx
// Allocate TMEM columns
tcgen05.alloc.cta_group::1.sync.aligned  tmem_addr, num_cols;

// MMA: inputs from SMEM, accumulator in TMEM
tcgen05.mma.cta_group::1.kind::f16  tmem_addr, desc_a, desc_b, idesc_c, idesc_d;

// 2-SM cooperative MMA
tcgen05.mma.cta_group::2.kind::f16  tmem_addr, desc_a, desc_b, idesc_c, idesc_d;

// Load TMEM to registers
tcgen05.ld.sync.aligned.32x32b.x1  {regs}, [tmem_addr];

// Store registers to TMEM
tcgen05.st.sync.aligned.32x32b.x1  [tmem_addr], {regs};

// Copy SMEM to TMEM
tcgen05.cp.sync.aligned  [tmem_addr], [smem_addr], num_bytes;

// Critical fence between TMA completion and MMA
tcgen05.fence::after_thread_sync;

// Deallocate TMEM (MUST before kernel exit)
tcgen05.dealloc.cta_group::1.sync.aligned  tmem_addr, num_cols;
```

## NVFP4 Conversion Instructions

```ptx
// Convert two packed FP4 values to two FP16 values
cvt.rn.f16x2.e2m1x2  result_f16x2, packed_fp4;

// Byte unpacking (faster than bitwise extraction)
mov.b32  {byte0, byte1, byte2, byte3}, packed_word;
```

## Cache Control for Memory-Bound Kernels

```ptx
// Streaming data (use once): bypass L1
ld.global.L1::no_allocate.v2.u64  {r0, r1}, [addr];

// Reused data: keep in L1
ld.global.L1::evict_last.v2.u64  {r0, r1}, [addr];

// Wide vectorized loads
ld.global.v4.u64  {r0, r1, r2, r3}, [addr];  // 256-bit
```

## Cluster Launch Control

```ptx
// Query for next tile (persistent kernel loop)
clusterlaunchcontrol.try_cancel  {clc_id};
// Returns valid tile_id or decline (all work done)
```

## TMA (Tensor Memory Accelerator)

```ptx
// Bulk tensor copy: global → shared
cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
    [smem_ptr], [tensorMap, {x, y}], [mbarrier];

// Multicast to cluster SMs
cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast
    [smem_ptr], [tensorMap, {x, y}], [mbarrier], multicast_mask;
```

## Related
- [cuda-cpp](cuda-cpp.md) — Inline PTX in CUDA C++
- [tcgen05-mma](../hardware/tcgen05-mma.md) — MMA instruction details
- [nvfp4](../hardware/nvfp4.md) — FP4 format details
