---
id: technique-kernel-fusion
title: "Kernel Fusion"
type: technique
architectures: [sm100, sm90]
tags: [kernel-fusion, fused-kernel, tmem]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-tmem]
related: [kernel-fused-moe, kernel-nvfp4-gemm, technique-epilogue-fusion]
sources: [contest-gpumode-p3, contest-flashinfer-track-a, blog-tflops-gap-fp4-moe]
blackwell_relevance: "TMEM enables multi-accumulator fusion (gate+up dual GEMM) without register pressure; technique valuable on both architectures."
---

## 本工作台的语义修正

减少中间tensor与launch的可测目标可作为候选方向。

TMEM总256列错误，应512；2x256 accumulator无scale余量；代码未同步释放；融合在Hopper也成立；框架固定launch数非架构事实。

正确实现按以下契约落实：预算整图中间读写，完整数值/alias契约，真实wrapper替代分配伪码。具体同步、数值和映射参考见[已核对的技术契约](../../../../references/kernelwiki-semantic-corrections.md)。

以下上游叙述为历史来源材料；上述冲突处以本修正为准。被指出的 CUDA/PTX/Triton 片段不能直接执行或标为数值通过；其历史性能不计作本次实测。

## 保留的上游叙述


# Kernel Fusion

## Overview

Kernel fusion combines multiple operations into a single kernel launch, eliminating intermediate global memory roundtrips. Critical for MoE and attention pipelines where 5-7 sequential launches each incur latency, synchronization, and memory traffic overhead.

## Examples

### Fused Gate-Up Dual GEMM + SwiGLU
```cuda
// Instead of: gate_gemm → up_gemm → silu → multiply (4 kernels)
// Fused: single kernel with two TMEM accumulators
__global__ void fused_gate_up_silu(...) {
    uint32_t tmem_gate = tmem_alloc(256);
    uint32_t tmem_up = tmem_alloc(256);

    for (int k = 0; k < K; k += BLOCK_K) {
        tcgen05_mma(x_smem, w_gate_smem, tmem_gate);
        tcgen05_mma(x_smem, w_up_smem, tmem_up);
    }

    float g = tmem_load(tmem_gate);
    float u = tmem_load(tmem_up);
    output = (g / (1.0f + expf(-g))) * u;  // SwiGLU fused
}
```

### MoE Fusion Progression
- **vLLM (7 kernels)**: softmax → topk → dispatch → gate → up → silu_mul → down+combine
- **SGLang (5 kernels)**: router+topk → dispatch → fused gate-up-silu → down → combine
- **Ideal (1-2 kernels)**: all ops in one launch, saves 21.9% activation memory traffic

## Constraints

- TMEM capacity limits how many accumulators can fuse (256 cols total)
- Register pressure on epilogue if fusing complex activations
- Fusion opportunities depend on dataflow shape (dependency graph must be DAG-compatible with CTA scope)

## Related
- [fused-moe](../kernels/fused-moe.md)
- [epilogue-fusion](epilogue-fusion.md)
