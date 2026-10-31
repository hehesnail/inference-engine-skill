---
id: hw-nvfp4
title: "NVFP4 and Block-Scaled Narrow Precision"
type: hardware
architectures: [sm100, sm100a]
tags: [nvfp4, fp4, block-scale, fp8, fp6]
confidence: source-reported
related: [technique-fine-grained-quantization, kernel-nvfp4-gemm, kernel-nvfp4-gemv, hw-tcgen05-mma]
sources: [doc-nvidia-tuning-guide, contest-gpumode-p1, contest-gpumode-p2, blog-yue-nvfp4]
aliases: [NVFP4, E2M1, "FP4 E2M1", "nv_float4"]
---

## 本工作台的语义修正

E2M1 数据、每16元素 E4M3 block scale、全局 FP32 scale 的格式契约可作为候选方向。

格式解释可保留；吞吐倍数未固定 SKU/稀疏性/时钟，误差更低非普遍保证；转换 PTX 需核对目标 ISA 和 rounding 支持。

正确实现按以下契约落实：绑定精确 checkpoint packing/scales、合法 scale 类型和参考反量化；历史吞吐标 source-reported。具体同步、数值和映射参考见[已核对的技术契约](../../../../references/kernelwiki-semantic-corrections.md)。

以下上游叙述为历史来源材料；上述冲突处以本修正为准。被指出的 CUDA/PTX/Triton 片段不能直接执行或标为数值通过；其历史性能不计作本次实测。

## 保留的上游叙述


## Overview

NVFP4 is NVIDIA's 4-bit floating-point format (E2M1) with block scaling, native to Blackwell tensor cores.

## Format Details

```
E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit
Representable values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6

Block scaling: every 16 FP4 elements share one FP8 E4M3 scale factor
Two-level: per-block E4M3 scale × per-tensor FP32 global scale

Quantization:   q_i = cast_FP4(x_i / (s_global * s_block))
Dequantization: x_hat_i = s_global * s_block * deq_FP4(q_i)
```

## tcgen05 Variants for FP4

| Variant | Description | Throughput vs Hopper |
|---|---|---|
| `tcgen05.mma.mxf4.block_scale` | MX FP4 with block scaling | **4×** |
| `tcgen05.mma.mxf4nvf4.block_scale` | NVFP4 + MX FP4 flexible scaling | **4×** |

## PTX for FP4 Conversion

```ptx
// Convert two FP4 values to two FP16 values
cvt.rn.f16x2.e2m1x2 result, packed_fp4;

// Byte unpacking (faster than bitwise extraction)
mov.b32 {tmp0, tmp1, tmp2, tmp3}, packed_data;
```

## NVFP4 vs MXFP4

| Aspect | NVFP4 | MXFP4 |
|---|---|---|
| Scale format | E4M3 (fractional) | UE8M0 (power-of-2 only) |
| Block size | 16 elements | 32 elements |
| Scale precision | Non-power-of-2 | Power-of-2 only |
| Quantization error | Lower | Higher |

## Related
- [fine-grained-quantization](../techniques/fine-grained-quantization.md) — Scaling strategies
- [nvfp4-gemm](../kernels/nvfp4-gemm.md) — NVFP4 GEMM kernel
- [nvfp4-gemv](../kernels/nvfp4-gemv.md) — NVFP4 GEMV kernel
