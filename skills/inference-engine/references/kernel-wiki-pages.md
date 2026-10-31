# KernelWiki 48篇技术页

先读[查询、来源边界与修正](kernel-knowledge.md)。这些页保留技术正文；引用到的PR、代码和大语料在完整服务器包中。

| 类别 | 技术页 |
|---|---|
| hardware | [Two-SM Cooperative MMA](../library/kernelwiki/wiki/hardware/2sm-cooperative.md) |
| hardware | [Cluster Launch Control (CLC)](../library/kernelwiki/wiki/hardware/clc.md) |
| hardware | [mbarrier (Memory Barrier Primitives)](../library/kernelwiki/wiki/hardware/mbarrier.md) |
| hardware | [NVFP4 and Block-Scaled Narrow Precision](../library/kernelwiki/wiki/hardware/nvfp4.md) |
| hardware | [Programmatic Dependent Launch / Grid Dependency Control](../library/kernelwiki/wiki/hardware/pdl-gdc.md) |
| hardware | [tcgen05.mma — Blackwell MMA Instruction](../library/kernelwiki/wiki/hardware/tcgen05-mma.md) |
| hardware | [Tensor Memory Accelerator (TMA)](../library/kernelwiki/wiki/hardware/tma.md) |
| hardware | [Tensor Memory (TMEM)](../library/kernelwiki/wiki/hardware/tmem.md) |
| kernels | [DeepGEMM — FP8 GEMM with Fine-Grained Scaling](../library/kernelwiki/wiki/kernels/deepgemm.md) |
| kernels | [FlashAttention-4](../library/kernelwiki/wiki/kernels/flash-attention-4.md) |
| kernels | [FlashMLA — Multi-head Latent Attention](../library/kernelwiki/wiki/kernels/flashmla.md) |
| kernels | [FP8 Block-Scale GEMM](../library/kernelwiki/wiki/kernels/fp8-block-scale-gemm.md) |
| kernels | [Fused MoE — FP8 Block-Scale Routing + Dual GEMM](../library/kernelwiki/wiki/kernels/fused-moe.md) |
| kernels | [Gated Delta Net — Linear Attention](../library/kernelwiki/wiki/kernels/gated-delta-net.md) |
| kernels | [Gated Dual GEMM (Gate-Up + SwiGLU Fusion)](../library/kernelwiki/wiki/kernels/gated-dual-gemm.md) |
| kernels | [Grouped GEMM for MoE](../library/kernelwiki/wiki/kernels/grouped-gemm.md) |
| kernels | [Native Sparse Attention (NSA)](../library/kernelwiki/wiki/kernels/nsa.md) |
| kernels | [NVFP4 GEMM — 4-bit Floating Point Matrix Multiply](../library/kernelwiki/wiki/kernels/nvfp4-gemm.md) |
| kernels | [NVFP4 Batched GEMV](../library/kernelwiki/wiki/kernels/nvfp4-gemv.md) |
| kernels | [Sparse MLA (DeepSeek V3.2)](../library/kernelwiki/wiki/kernels/sparse-mla.md) |
| languages | [CUDA C++ for Blackwell Kernels](../library/kernelwiki/wiki/languages/cuda-cpp.md) |
| languages | [CuTe DSL for Blackwell](../library/kernelwiki/wiki/languages/cute-dsl.md) |
| languages | [PTX Instructions for SM100](../library/kernelwiki/wiki/languages/ptx-sm100.md) |
| languages | [Triton on Blackwell](../library/kernelwiki/wiki/languages/triton-blackwell.md) |
| migration | [Register Accumulators to TMEM](../library/kernelwiki/wiki/migration/register-to-tmem.md) |
| migration | [Migrating from wgmma to tcgen05](../library/kernelwiki/wiki/migration/wgmma-to-tcgen05.md) |
| patterns | [Not Reaching Peak FLOPS](../library/kernelwiki/wiki/patterns/compute-bound.md) |
| patterns | [Low SM Utilization](../library/kernelwiki/wiki/patterns/low-sm-utilization.md) |
| patterns | [Memory Bandwidth Bound](../library/kernelwiki/wiki/patterns/memory-bound.md) |
| patterns | [MoE Expert Load Imbalance](../library/kernelwiki/wiki/patterns/moe-load-imbalance.md) |
| patterns | [Pipeline Stalls](../library/kernelwiki/wiki/patterns/pipeline-stalls.md) |
| patterns | [Register Pressure — Low Occupancy](../library/kernelwiki/wiki/patterns/register-pressure.md) |
| patterns | [Tail Effect — Last Wave Underutilization](../library/kernelwiki/wiki/patterns/tail-effect.md) |
| techniques | [PTX Cache Policy Differentiation](../library/kernelwiki/wiki/techniques/cache-policy.md) |
| techniques | [Chunk-Based Parallelism for Linear Attention](../library/kernelwiki/wiki/techniques/chunk-parallelism.md) |
| techniques | [Double/Multi-Buffering Patterns](../library/kernelwiki/wiki/techniques/double-buffering.md) |
| techniques | [Epilogue Fusion](../library/kernelwiki/wiki/techniques/epilogue-fusion.md) |
| techniques | [Fine-Grained FP8/FP4 Quantization](../library/kernelwiki/wiki/techniques/fine-grained-quantization.md) |
| techniques | [Kernel Fusion](../library/kernelwiki/wiki/techniques/kernel-fusion.md) |
| techniques | [Persistent Kernels with CLC](../library/kernelwiki/wiki/techniques/persistent-kernels.md) |
| techniques | [Ping-Pong Scheduling](../library/kernelwiki/wiki/techniques/ping-pong-scheduling.md) |
| techniques | [Software Pipelining and Multi-Stage Buffering](../library/kernelwiki/wiki/techniques/pipeline-stages.md) |
| techniques | [Register Budgeting for Occupancy](../library/kernelwiki/wiki/techniques/register-budgeting.md) |
| techniques | [Software-Emulated Exponential](../library/kernelwiki/wiki/techniques/software-exp.md) |
| techniques | [Shared Memory Swizzling](../library/kernelwiki/wiki/techniques/swizzling.md) |
| techniques | [Tile Scheduling Strategies](../library/kernelwiki/wiki/techniques/tile-scheduling.md) |
| techniques | [Wide Vectorized Loads and Cache Policies](../library/kernelwiki/wiki/techniques/vectorized-loads.md) |
| techniques | [Warp Specialization on Blackwell](../library/kernelwiki/wiki/techniques/warp-specialization.md) |
