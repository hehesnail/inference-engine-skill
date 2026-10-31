---
id: pattern-compute-bound
title: "Not Reaching Peak FLOPS"
type: pattern
tags: [tcgen05, 2sm-cooperative, pipeline-stages, warp-specialization]
symptoms: [compute-bound, low-tensor-core-utilization, pipeline-stalls]
candidate_techniques: [hw-2sm-cooperative, technique-pipeline-stages, technique-warp-specialization, technique-epilogue-fusion, technique-software-exp]
related: [pattern-low-sm-utilization, pattern-register-pressure]
sources: [doc-nvidia-tuning-guide, blog-tcgen05-tutorial, blog-flash-attention-4]
---

## 本工作台的语义修正

以算术强度和实测吞吐定位compute限制可作为候选方向。

70%TC/低带宽不足以独立证明compute-bound；2SM模式不把整个GPU峰值翻倍；无stall/普遍收益没有证据。

正确实现按以下契约落实：结合roofline、指令/依赖和形状，按相同datatype/稀疏模式取理论峰值；A/B归因。具体同步、数值和映射参考见[已核对的技术契约](../../../../references/kernelwiki-semantic-corrections.md)。

以下上游叙述为历史来源材料；上述冲突处以本修正为准。被指出的 CUDA/PTX/Triton 片段不能直接执行或标为数值通过；其历史性能不计作本次实测。

## 保留的上游叙述


## Symptom

Tensor core utilization below 70%. Memory bandwidth is not saturated. Kernel is compute-bound but not reaching peak FLOPS.

## Likely Causes

1. **Pipeline bubbles**: MMA stalled waiting for data from TMA
2. **Non-matmul overhead**: Softmax, activation functions, reductions consuming cycles
3. **Single-SM MMA tiles too small**: Not fully utilizing available compute
4. **Epilogue blocking mainloop**: TMEM reads blocking next MMA

## Candidate Techniques

| Technique | Effect |
|---|---|
| [2-SM cooperative](../hardware/2sm-cooperative.md) | Double effective MMA tile (m256×n256), 2× compute per cycle |
| [Pipeline stages](../techniques/pipeline-stages.md) | Overlap TMA load with MMA compute |
| [Warp specialization](../techniques/warp-specialization.md) | Dedicated warps for TMA/MMA/epilogue, no stalls |
| [Epilogue fusion](../techniques/epilogue-fusion.md) | Overlap epilogue with next tile's MMA |
| [Software exponential](../techniques/software-exp.md) | Distribute non-matmul ops across FMA units (FA4) |

## Example: FlashAttention-4

```
// Problem: Blackwell doubles tensor core throughput but SFU count unchanged
// SFU bottleneck: exp() for softmax
//
// Solution: Software 2^x via Cody-Waite + Horner polynomial
// Distributes across FMA units, multiplying exponential throughput
// Result: 1605 TFLOPS (71% utilization) on B200
```

## Caveats
- 2-SM cooperative requires cluster configuration and identical SMEM layouts
- Pipeline depth tuning is workload-dependent (3-5 stages typical)
- Software-emulated transcendentals trade accuracy for throughput
