---
id: pattern-tail-effect
title: "Tail Effect — Last Wave Underutilization"
type: pattern
tags: [persistent-kernel, clc, tile-scheduling]
symptoms: [tail-effect, low-sm-utilization, wave-quantization]
candidate_techniques: [technique-persistent-kernels, hw-clc, technique-tile-scheduling]
related: [pattern-low-sm-utilization]
sources: [doc-nvidia-tuning-guide, blog-tcgen05-tutorial, pr-cutlass-2161]
---

## 本工作台的语义修正

last-wave不均衡及granularity成本分析可作为候选方向。

150tiles/142SM最后8tiles不会因CLC就同时占142SM；不能把多项教程变化的整体146%归因tail。

正确实现按以下契约落实：保留可复算wave占用模型，splitK需额外归约成本；实际profiling区分任务大小和调度成本。具体同步、数值和映射参考见[已核对的技术契约](../../../../references/kernelwiki-semantic-corrections.md)。

以下上游叙述为历史来源材料；上述冲突处以本修正为准。被指出的 CUDA/PTX/Triton 片段不能直接执行或标为数值通过；其历史性能不计作本次实测。

## 保留的上游叙述


## Symptom

Performance drops for problem sizes where total_tiles % num_SMs != 0. The last wave of tiles runs with many SMs idle.

## Likely Causes

1. **Wave quantization**: Grid of N tiles on M SMs takes ceil(N/M) waves; last wave may use only N%M SMs
2. **Static assignment**: stride-by-gridDim leaves remainder tiles on few SMs
3. **Non-persistent launch**: each kernel launch has fixed grid, no dynamic rebalancing

## Candidate Techniques

| Technique | Effect |
|---|---|
| [CLC](../hardware/clc.md) | Hardware dynamic scheduling, SMs grab tiles on-demand |
| [Persistent kernels](../techniques/persistent-kernels.md) | SM-count grid, iterate over tiles, no wave boundary |
| [Tile scheduling](../techniques/tile-scheduling.md) | Raster order, swizzle patterns for better distribution |

## Example

```
// B200: 142 SMs
// Problem: 150 tiles
// Without CLC: 2 waves (142 + 8), last wave uses only 8 SMs (5.6%)
// With CLC: single persistent wave, all 142 SMs stay busy
//
// Impact: 86% → 98% of cuBLAS (tcgen05 tutorial data)
```

## Caveats
- Only significant for moderate tile counts (< 4× SM count)
- For very large problems, tail effect is amortized across many waves
- CLC only on SM100 datacenter
