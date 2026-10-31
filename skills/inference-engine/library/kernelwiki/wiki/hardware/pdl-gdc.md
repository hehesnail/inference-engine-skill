---
id: hw-pdl-gdc
title: "Programmatic Dependent Launch / Grid Dependency Control"
type: hardware
architectures: [sm100, sm100a, sm90]
tags: [pdl, gdc]
confidence: source-reported
related: [technique-persistent-kernels, hw-clc]
sources: [doc-nvidia-tuning-guide, pr-cutlass-2161, doc-cutlass-changelog-sm100]
aliases: [PDL, GDC, "programmatic dependent launch", "grid dependency control"]
blackwell_relevance: "PDL available on Hopper but enabled by default on Blackwell SM100."
---


# Programmatic Dependent Launch / Grid Dependency Control

Local correction, 2026-10-31. The original page confused the producer trigger with the consumer wait and generalized a library default to all Blackwell launches. Original source is retained under upstream-reference/hardware/pdl-gdc.md.

PDL allows a secondary kernel to do independent work before its predecessor completes. It requires the supported launch configuration; Blackwell hardware alone does not enable it for every ordinary launch.

- Launch the secondary with cudaLaunchAttributeProgrammaticStreamSerialization and programmaticStreamSerializationAllowed = 1, or the corresponding supported graph dependency.
- The primary uses cudaTriggerProgrammaticLaunchCompletion to release its launch dependency; all relevant primary blocks must reach the required point.
- Before consuming primary output, the secondary uses cudaGridDependencySynchronize or an equivalent verified synchronization protocol.
- Opportunistic overlap is not guaranteed. Correctness must not depend on both kernels making concurrent progress.
- Check the actual engine/kernel wrapper and CUDA version; a CUTLASS or other library's default is local to that implementation.

Sources: [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html), [CUTLASS dependent launches](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/dependent_kernel_launch.md).

Related: [persistent kernels](../techniques/persistent-kernels.md), [CLC](clc.md).
