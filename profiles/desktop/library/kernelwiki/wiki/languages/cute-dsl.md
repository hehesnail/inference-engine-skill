---
id: lang-cute-dsl
title: "CuTe DSL for Blackwell"
type: language
tags: [cute-dsl, tcgen05, tmem, tma]
related: [hw-tcgen05-mma, hw-tmem, kernel-flash-attention-4, doc-cutlass-blackwell]
sources: [doc-cutlass-blackwell, blog-colfax-cutlass, blog-flash-attention-4]
reproducibility: snippet
architectures: [sm100, sm100a]
confidence: source-reported
---

## 本工作台的语义修正

CuTe layout代数、版本固定的SM100原子组合可作为候选方向。

cute_kernel、SM100_MMA_F16BF16_SS、make_tmacopy等不是通用现行API；stage正文所列9个CuTe artifact链接实际存在，并非材料缺失，仍须固定版本核对签名。

正确实现按以下契约落实：按实际依赖版本用 cutlass.cute.nvgpu 的 atoms；优先读取本页链接的锁定完整实现，不能把教学伪API当可构建代码。具体同步、数值和映射参考见[已核对的技术契约](../../../../references/kernelwiki-semantic-corrections.md)。

以下上游叙述为历史来源材料；上述冲突处以本修正为准。被指出的 CUDA/PTX/Triton 片段不能直接执行或标为数值通过；其历史性能不计作本次实测。

## 保留的上游叙述


## Overview

CuTe (CUDA Templates) DSL is the primary abstraction layer in CUTLASS 4.5.0 for Blackwell kernels. FlashAttention-4 was implemented entirely in CuTe-DSL (Python variant), achieving 20-30× faster compilation than C++ templates.

## SM100 MMA Atoms

```python
# CuTe-DSL: SM100 MMA atom for BF16
from cutlass.cute import *

# 1-SM MMA: m128 x n256 x k16
mma_atom = SM100_MMA_F16BF16_SS  # inputs from shared memory
# Accumulator goes to TMEM automatically

# 2-SM MMA: m256 x n256 x k16
mma_atom_2sm = SM100_MMA_F16BF16_SS_2SM
```

## TMEM as CuTe Locale

```python
# TMEM allocation in CuTe
tmem_tensor = make_tensor(
    make_tmem_ptr(tmem_addr),
    make_layout(make_shape(128, 256))  # rows x cols
)

# Copy TMEM → registers for epilogue
copy(tmem_tensor, reg_tensor)  # tcgen05.ld under the hood
```

## TMA Copy Atoms

```python
# TMA bulk copy: global → shared
tma_copy = SM100_TMA_LOAD_2D

# Setup TMA descriptor
tma_desc = make_tma_copy(
    tma_copy,
    global_tensor,
    smem_layout,
    tile_shape,
    cluster_shape
)
```

## Warp-Specialized Kernel Skeleton

```python
@cute_kernel
def blackwell_gemm(A, B, C):
    # Warp 0: TMA producer
    if warp_id == 0:
        for stage in pipeline:
            tma_copy(A_tile, smem_A[stage])
            tma_copy(B_tile, smem_B[stage])
            arrive(mbarrier[stage])

    # Warp 1: MMA consumer
    elif warp_id == 1:
        tmem = alloc_tmem(256)  # columns
        for stage in pipeline:
            wait(mbarrier[stage])
            fence_after_thread_sync()
            mma(smem_A[stage], smem_B[stage], tmem)
        signal_epilogue()

    # Warps 2+: Epilogue
    else:
        wait_epilogue()
        regs = load_tmem(tmem)
        store_global(C, regs)
        dealloc_tmem(tmem)
```

## Why CuTe-DSL for Blackwell

1. **20-30× faster compilation** than C++ CUTLASS templates
2. Python-native: easier to iterate and debug
3. Same performance as hand-written C++ (FlashAttention-4: 1605 TFLOPS)
4. First-class TMEM and tcgen05 support in CUTLASS 4.5.0
5. Automatic layout computation and swizzle handling

## Full Examples (verbatim upstream code shipped locally)

The following CuTe DSL files ship **verbatim** in this repository under `artifacts/prs/cutlass/` (pinned at each PR's merge SHA). Open them with `python3 scripts/get_page.py <pr-id> --include-code` or read them directly.

| File | Purpose | Size |
|---|---|---|
| [`artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_0.py`](https://github.com/mit-han-lab/KernelWiki/blob/76d27b56f804e7e7295d4c570e1e5d7eef4b0a75/artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_0.py) | Step 0 — FP16 GEMM baseline | 447 lines |
| [`artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_1.py`](https://github.com/mit-han-lab/KernelWiki/blob/76d27b56f804e7e7295d4c570e1e5d7eef4b0a75/artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_1.py) | Step 1 — 2CTA MMA + TMA multicast | 535 lines |
| [`artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_2.py`](https://github.com/mit-han-lab/KernelWiki/blob/76d27b56f804e7e7295d4c570e1e5d7eef4b0a75/artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_2.py) | Step 2 — Warp specialization (TMA / MMA / epilogue warps) | 679 lines |
| [`artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_3.py`](https://github.com/mit-han-lab/KernelWiki/blob/76d27b56f804e7e7295d4c570e1e5d7eef4b0a75/artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_3.py) | Step 3 — Static persistent tile scheduler | 769 lines |
| [`artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_4.py`](https://github.com/mit-han-lab/KernelWiki/blob/76d27b56f804e7e7295d4c570e1e5d7eef4b0a75/artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_4.py) | Step 4 — Preferred + dynamic clusters | 1065 lines |
| [`artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_5.py`](https://github.com/mit-han-lab/KernelWiki/blob/76d27b56f804e7e7295d4c570e1e5d7eef4b0a75/artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_5.py) | Step 5 — TMA prefetch | 919 lines |
| [`artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_6.py`](https://github.com/mit-han-lab/KernelWiki/blob/76d27b56f804e7e7295d4c570e1e5d7eef4b0a75/artifacts/prs/cutlass/PR-3106/key-files/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_6.py) | Step 6 — Programmatic Dependent Launch (PDL) | 1002 lines |
| [`artifacts/prs/cutlass/PR-2881/key-files/examples/python/CuTeDSL/blackwell/dense_gemm_persistent_prefetch.py`](https://github.com/mit-han-lab/KernelWiki/blob/76d27b56f804e7e7295d4c570e1e5d7eef4b0a75/artifacts/prs/cutlass/PR-2881/key-files/examples/python/CuTeDSL/blackwell/dense_gemm_persistent_prefetch.py) | Persistent GEMM with prefetch | full |
| [`artifacts/prs/cutlass/PR-3021/key-files/python/CuTeDSL/cutlass/cute/arch/clc.py`](https://github.com/mit-han-lab/KernelWiki/blob/76d27b56f804e7e7295d4c570e1e5d7eef4b0a75/artifacts/prs/cutlass/PR-3021/key-files/python/CuTeDSL/cutlass/cute/arch/clc.py) | CLC (Cluster Launch Control) Python binding | full |

The `fp16_gemm_{0..6}.py` series from `examples/python/CuTeDSL/blackwell/tutorial_gemm/` in NVIDIA/cutlass PR-3106 is the authoritative CuTe DSL learning path: it walks from a naive FP16 GEMM baseline through 2CTA MMA with TMA multicast, warp specialization, static persistent scheduling, preferred / dynamic clusters, TMA prefetch, and ends with Programmatic Dependent Launch (PDL). Reading them in order is the recommended on-ramp.

## Related
- [tcgen05-mma](../hardware/tcgen05-mma.md) — Underlying MMA instruction
- [flash-attention-4](../kernels/flash-attention-4.md) — CuTe-DSL implementation
- [CUTLASS Blackwell docs](https://github.com/mit-han-lab/KernelWiki/blob/76d27b56f804e7e7295d4c570e1e5d7eef4b0a75/sources/docs/nvidia-cutlass-blackwell.md) — Official reference
