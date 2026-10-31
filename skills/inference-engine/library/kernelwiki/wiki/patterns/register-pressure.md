---
id: pattern-register-pressure
title: "Register Pressure — Low Occupancy"
type: pattern
tags: [tmem, register-reuse, warp-specialization]
symptoms: [register-pressure, low-occupancy, register-spilling]
candidate_techniques: [hw-tmem, technique-warp-specialization, migration-register-to-tmem]
related: [pattern-compute-bound, hw-tmem]
sources: [doc-nvidia-tuning-guide, blog-tcgen05-tutorial, pr-vllm-16032]
---

## 本工作台的语义修正

寄存器/占用率/spill 联合诊断可作为候选方向。

TMEM只移accumulator，epilogue仍需regs；warp specialization在Hopper已存在，不能标SM100-only；只按register数反推occupancy不完整。

正确实现按以下契约落实：用编译器资源和运行occupancy交叉验证，保留SMEM/TMEM与CTA限制。具体同步、数值和映射参考见[已核对的技术契约](../../../../references/kernelwiki-semantic-corrections.md)。

以下上游叙述为历史来源材料；上述冲突处以本修正为准。被指出的 CUDA/PTX/Triton 片段不能直接执行或标为数值通过；其历史性能不计作本次实测。

## 保留的上游叙述


## Symptom

Occupancy below target due to high register usage per thread. Nsight Compute shows register spilling to local memory.

## Likely Causes

1. **Accumulator registers**: On Hopper, large MMA tiles consume many registers for accumulators
2. **Epilogue state**: Data transformation in epilogue requires additional registers
3. **Complex control flow**: Many live variables across branches

## Candidate Techniques

| Technique | Applicability | Effect |
|---|---|---|
| [TMEM](../hardware/tmem.md) | SM100 only | Moves accumulators to dedicated 256KB memory |
| [Warp specialization](../techniques/warp-specialization.md) | SM100+ | Different warps handle different roles, reducing per-warp register needs |
| [Register-to-TMEM migration](../migration/register-to-tmem.md) | SM90→SM100 | Systematic approach to moving accumulators off registers |

## Blackwell Solution

```
// Hopper: 64×256 MMA tile accumulator = 64*256*4 bytes in registers per warp group
//   → ~128 registers per thread just for accumulators
//
// Blackwell: TMEM holds accumulators
//   → 0 registers for accumulators
//   → ~128 registers freed for other work or higher occupancy
//
// TMEM: 256 KB per SM, 128 rows × 512 columns × 32-bit
// Largest 1-SM MMA uses half → double-buffering possible
```

## Caveats
- TMEM only available on SM100 datacenter (not SM120 consumer)
- TMEM requires explicit alloc/dealloc lifecycle
- TMEM→register transfer adds latency (offset by freeing registers)
