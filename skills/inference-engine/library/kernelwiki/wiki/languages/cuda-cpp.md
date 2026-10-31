---
id: lang-cuda-cpp
title: "CUDA C++ for Blackwell Kernels"
type: language
tags: [cuda-cpp, ptx, tcgen05, tmem]
related: [lang-ptx, hw-tcgen05-mma, hw-tmem, blog-tcgen05-tutorial]
sources: [blog-tcgen05-tutorial, doc-nvidia-tuning-guide, blog-yue-nvfp4]
reproducibility: snippet
architectures: [sm100, sm100a]
confidence: source-reported
---

## 本工作台的语义修正

低层CUDA与PTX边界、编译目标声明可作为候选方向。

alloc输出/collective范围、MMA描述符、TMEM load与内联寄存器约束多处不能编译；省略完整生命周期却称compilable。

正确实现按以下契约落实：只保留固定官方atom调用入口；教学流程用无语言伪码，不作为可构建实现。具体同步、数值和映射参考见[已核对的技术契约](../../../../references/kernelwiki-semantic-corrections.md)。

以下上游叙述为历史来源材料；上述冲突处以本修正为准。被指出的 CUDA/PTX/Triton 片段不能直接执行或标为数值通过；其历史性能不计作本次实测。

## 保留的上游叙述


## Overview

Plain CUDA C++ with inline PTX is used for hand-optimized Blackwell kernels. The tcgen05 tutorial achieved 98% of cuBLAS performance using this approach.

## tcgen05 via Inline PTX

```cuda
// Allocate TMEM
__device__ void tmem_alloc(uint32_t* addr, uint32_t num_cols) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned %0, %1;"
        : "=r"(*addr) : "r"(num_cols)
    );
}

// Issue MMA (single thread, typically warp 1 lane 0)
// idesc_c/idesc_d: immediate descriptors for accumulator C and output D
__device__ void tcgen05_mma(uint32_t tmem_addr,
                             uint64_t desc_a, uint64_t desc_b,
                             uint32_t idesc_c, uint32_t idesc_d) {
    asm volatile(
        "tcgen05.mma.cta_group::1.kind::f16"
        " %0, %1, %2, %3, %4;"
        :: "r"(tmem_addr), "l"(desc_a), "l"(desc_b),
           "r"(idesc_c), "r"(idesc_d)
    );
}

// Load TMEM to registers
__device__ void tmem_load(float* dst, uint32_t tmem_addr, int cols) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x1 {%0}, [%1];"
        : "=r"(*dst) : "r"(tmem_addr)
    );
}

// Deallocate TMEM (MUST do before kernel exit)
__device__ void tmem_dealloc(uint32_t addr, uint32_t num_cols) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned %0, %1;"
        :: "r"(addr), "r"(num_cols)
    );
}
```

## mbarrier Synchronization

```cuda
// TMA-MMA synchronization via mbarrier
// expected_bytes: total bytes the TMA will deliver to this stage
__device__ void mbarrier_arrive(uint64_t* mbar, uint32_t expected_bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(mbar)),
           "r"(expected_bytes)
    );
}

__device__ void mbarrier_wait(uint64_t* mbar, int phase) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WAIT_LOOP:\n"
        "  mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
        "  @!p bra WAIT_LOOP;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(mbar)),
           "r"(phase)
    );
}
```

## Warp Role Dispatch

```cuda
__global__ void blackwell_gemm_kernel(...) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_id == 0 && lane_id == 0) {
        // TMA producer: issue cp.async.bulk.tensor
        tma_producer_loop(...);
    } else if (warp_id == 1 && lane_id == 0) {
        // MMA consumer: issue tcgen05.mma
        mma_consumer_loop(...);
    } else if (warp_id >= 2) {
        // Epilogue: read TMEM, write to global
        epilogue_loop(...);
    }
}
```

## Related
- [ptx-sm100](ptx-sm100.md) — PTX instruction reference
- [tcgen05 tutorial](../../sources/blogs/tcgen05-tutorial.md) — Step-by-step guide
