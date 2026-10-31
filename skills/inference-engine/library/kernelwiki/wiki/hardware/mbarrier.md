---
id: hw-mbarrier
title: "mbarrier (Memory Barrier Primitives)"
type: hardware
architectures: [sm100, sm100a, sm90, sm90a]
tags: [mbarrier, tma]
confidence: source-reported
related: [hw-tma, hw-tcgen05-mma, technique-warp-specialization, technique-pipeline-stages]
sources: [doc-ptx-isa-sm100, blog-tcgen05-tutorial, doc-nvidia-tuning-guide]
aliases: [mbarrier, "memory barrier", "mbar"]
blackwell_relevance: "mbarrier is the primary synchronization primitive between TMA producers and tcgen05 consumers on Blackwell. Phase/parity tracking critical for pipelined kernels."
---

## 本工作台的语义修正

区分到达计数与异步传输字节计数、逐 slot parity可作为候选方向。

stage 已修每 slot phase、双计数归零语义；仍称 Hopper 引入，mbarrier 实为 sm80；需区分 TMA complete_tx 与线程 arrive，不允许 MMA issue 后立即释放 SMEM。

正确实现按以下契约落实：保留 stage 修正；修架构为 Ampere sm80，TMA/mbarrier complete_tx 的具体组合从 sm90 起；补 full/free 生命周期。具体同步、数值和映射参考见[已核对的技术契约](../../../../references/kernelwiki-semantic-corrections.md)。

以下上游叙述为历史来源材料；上述冲突处以本修正为准。被指出的 CUDA/PTX/Triton 片段不能直接执行或标为数值通过；其历史性能不计作本次实测。

## 保留的上游叙述


# mbarrier

## Overview

mbarriers are 64-bit shared memory primitives used for producer/consumer synchronization between asynchronous hardware units (TMA, tcgen05) and SM threads. Introduced on Hopper, essential for Blackwell warp-specialized kernels.

## Key Operations

```ptx
// Initialize: set expected arrival count
mbarrier.init.shared.b64 [mbar_addr], num_arrivals;

// Producer: arrive on barrier (decrements expected count)
mbarrier.arrive.shared.b64 _, [mbar_addr];

// Producer (with byte expectation): used by TMA
mbarrier.arrive.expect_tx.shared.b64 _, [mbar_addr], expected_bytes;
// After this, TMA hardware completes the transaction by arriving with
// the transferred byte count, and the mbarrier flips parity.

// Consumer: wait for parity flip
mbarrier.try_wait.parity.shared.b64 p, [mbar_addr], phase;
@!p bra WAIT_LOOP;
```

## Phase/Parity Semantics

An mbarrier phase completes when both its pending-arrival count and transaction count reach zero. Each reused stage has its own tracked parity. Consumers track their expected phase and wait for it:

```cuda
int phase[NUM_STAGES] = {0};
for (int k = 0; k < num_iterations; k++) {
    int stage = k % NUM_STAGES;

    // Wait for this stage's producer to complete
    mbarrier_wait_parity(&mbar[stage], phase[stage]);
    phase[stage] ^= 1;  // Flip for next use of this stage slot

    // Use data...
    consume_data(stage);

    // Arrive on next-stage barrier when done
    mbarrier_arrive(&buffer_free[stage]);
}
```

## TMA Integration

TMA directly signals mbarriers when transfer completes, avoiding manual polling:

```cuda
// Producer warp issues TMA with mbarrier target
if (lane_id == 0) {
    uint32_t tx_bytes = TILE_A_BYTES + TILE_B_BYTES;
    mbarrier_arrive_expect_tx(&mbar[stage], tx_bytes);

    // TMA completion will fire the mbarrier automatically
    cp_async_bulk_tensor_2d(smem_A[stage], global_desc, x, y, &mbar[stage]);
    cp_async_bulk_tensor_2d(smem_B[stage], global_desc, x, y, &mbar[stage]);
}
// Do NOT manually arrive after TMA issue - it races with hardware
```

## Pitfalls

1. **Missing phase flip**: Reusing a stage slot without flipping parity causes stale arrivals to satisfy the new wait
2. **Mismatched init count**: If consumer expects N arrivals but producers only arrive M times, barrier never fires
3. **Manual arrive after async issue**: TMA/tcgen05 hardware arrives on completion — extra manual arrive causes double-count
4. **Cross-cluster mbarriers**: Use `shared::cluster` qualifier for cluster-level sync (2-SM cooperative MMA)

## Related
- [TMA](tma.md) — Primary producer for mbarriers
- [warp-specialization](../techniques/warp-specialization.md) — Uses mbarriers for warp role handoff

Local correction: stage parity is tracked per slot; this is illustrative pseudocode, not a complete producer/consumer protocol. Check tx bytes, expected arrivals, fences and buffer-free ownership in the actual kernel. [PTX phase completion](https://docs.nvidia.com/cuda/archive/12.0.1/pdf/ptx_isa_8.0.pdf).
