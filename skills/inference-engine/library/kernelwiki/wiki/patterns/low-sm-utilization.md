---
id: pattern-low-sm-utilization
title: "Low SM Utilization"
type: pattern
tags: [persistent-kernel, clc, tile-scheduling]
symptoms: [low-sm-utilization, tail-effect, load-imbalance]
candidate_techniques: [technique-persistent-kernels, technique-tile-scheduling, hw-clc]
related: [pattern-tail-effect, pattern-compute-bound]
sources: [doc-nvidia-tuning-guide, blog-tcgen05-tutorial, pr-cutlass-2161]
---

## 本工作台的语义修正

小grid、资源限制和负载不均衡分开诊断可作为候选方向。

CLC不能创造并行或完全消灭tail；persistent技术非sm90起源；普通grid已动态分派CTA。

正确实现按以下契约落实：区分可运行CTA不足与软件调度overhead；tile/splitK/批合并各按正确性和归约成本评估。具体同步、数值和映射参考见[已核对的技术契约](../../../../references/kernelwiki-semantic-corrections.md)。

以下上游叙述为历史来源材料；上述冲突处以本修正为准。被指出的 CUDA/PTX/Triton 片段不能直接执行或标为数值通过；其历史性能不计作本次实测。

## 保留的上游叙述


## Symptom

SM utilization below 60% despite sufficient occupancy. Nsight Compute shows idle SMs during portions of kernel execution.

## Likely Causes

1. **Tail effect**: Last wave of tiles leaves most SMs idle (see [tail-effect](tail-effect.md))
2. **Load imbalance**: Some tiles take longer than others (variable computation per tile)
3. **Static scheduling**: Fixed tile-to-SM assignment doesn't adapt to runtime conditions
4. **Grid too small**: Fewer threadblocks than SMs

## Candidate Techniques

| Technique | Applicability | Effect |
|---|---|---|
| [CLC](../hardware/clc.md) | SM100 only | Dynamic tile assignment, eliminates load imbalance |
| [Persistent kernels](../techniques/persistent-kernels.md) | SM90+ | Eliminates tail effect, one-time launch overhead |
| [Tile scheduling](../techniques/tile-scheduling.md) | SM90+ | Better L2 locality, reduce load variance |

## Examples

```
// tcgen05 tutorial progression:
// Without persistent/CLC: 86% of cuBLAS (some SMs idle at wave boundaries)
// With persistent + CLC:  98% of cuBLAS (all SMs stay busy)
```

## Caveats
- CLC only available on SM100 datacenter GPUs (not SM120 consumer)
- Persistent kernels complicate debugging and profiling
- For non-persistent kernels, ensure grid size >> SM count
