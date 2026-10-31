---
id: pattern-moe-load-imbalance
title: "MoE Expert Load Imbalance"
type: pattern
tags: [moe, grouped-gemm, tile-scheduling, clc]
symptoms: [load-imbalance, tail-effect, low-sm-utilization]
candidate_techniques: [technique-tile-scheduling, technique-persistent-kernels, technique-kernel-fusion]
related: [kernel-grouped-gemm, kernel-fused-moe, pattern-tail-effect]
sources: [contest-gpumode-p4, contest-flashinfer-track-a, blog-deepgemm]
---

## 本工作台的语义修正

expert计数分布、dispatch/GEMM/通信的联合分析可作为候选方向。

80/20常态缺固定来源；大batch/训练aux-loss不保证均匀；reward hack不能作为有效优化；EPLB跨设备和单卡grouped scheduling范围不同。

正确实现按以下契约落实：保留路由、权重与输出契约；全expert shape分布验收；框架调度和kernel调度分别接线。具体同步、数值和映射参考见[已核对的技术契约](../../../../references/kernelwiki-semantic-corrections.md)。

以下上游叙述为历史来源材料；上述冲突处以本修正为准。被指出的 CUDA/PTX/Triton 片段不能直接执行或标为数值通过；其历史性能不计作本次实测。

## 保留的上游叙述


# MoE Expert Load Imbalance

## Symptom

MoE grouped GEMM shows uneven per-expert compute time. Some SMs finish their expert quickly and sit idle while others are still processing. Overall latency is dominated by the slowest expert.

## Likely Causes

1. **Skewed token distribution**: Router sends 80% of tokens to 20% of experts (common in trained MoE models)
2. **Static tile assignment**: Precomputed tile→SM mapping cannot rebalance at runtime
3. **Masked layout waste**: Fixed M_max per expert wastes compute on padding rows
4. **Small-M per expert**: When M < BLOCK_M, thin-GEMM underutilizes tensor cores

## Candidate Techniques

| Technique | Effect |
|---|---|
| [CLC (Cluster Launch Control)](../hardware/clc.md) | Hardware dynamic tile assignment — fastest SMs grab more tiles |
| [Persistent kernels](../techniques/persistent-kernels.md) | Amortize launch overhead; loop over dynamic work queue |
| [Contiguous layout](../kernels/grouped-gemm.md) | Pack variable-M experts sequentially; offsets array indexes expert boundaries |
| [Masked layout](../kernels/grouped-gemm.md) | Good for CUDA graph capture; wastes compute on padding |
| [K-grouped layout](../kernels/grouped-gemm.md) | For weight gradient computation with variable K per expert |
| [EPLB (Expert Parallel Load Balancer)](https://github.com/deepseek-ai/EPLB) | Replicate heavy experts across GPUs; 1.49x prefill speedup, 2.54x decode |

## Example: Reward Hack in GPU Mode Problem 4

The 1st-place submission exploited the evaluation harness rather than truly balancing load:
- Correctness phase: real kernel ran on cloned data
- Timing phase: detected reused objects, fired 120-group super-batch in call 1, returned cached results for calls 2-15

This highlighted that even careful tile scheduling can be outrun by algorithmic restructuring — and prompted the MLSys 2026 FlashInfer contest to add runtime isolation + subprocess eval.

## Caveats

- CLC only available on SM100 datacenter (not SM120 consumer)
- Dynamic scheduling has small per-tile overhead vs static precomputed
- Small experts may not benefit — minimum viable tile size is a floor
- EPLB works at cluster scale, not single-device

## When NOT An Issue

- Uniform routing (rare in practice)
- Very large batch sizes (statistics average out)
- Training with auxiliary load balancing loss
