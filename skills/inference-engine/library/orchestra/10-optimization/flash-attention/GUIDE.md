# FlashAttention 与 attention backend

FlashAttention 通过分块与 online softmax 减少 HBM 中间读写；它不改变 attention 所需要的 mask、position 和 tensor 契约。先验证输入布局再研究 kernel 性能。

## 可复用的实现检查
PyTorch 常见 Q/K/V 布局为 BHSD，FlashAttention 接口常见 BSHD 或 packed/varlen 形式。不能把同一 tensor 不经转置直接复用。核对 heads/GQA、strides、dtype、head_dim、cu_seqlens、窗口、ALiBi、softmax scale 与 dropout；推理 dropout 应为0。

矩形 Q/K 的 causal 对齐尤其重要：增量 decode 的一个 query 应看到其所有合法历史 keys；FA2.1 起有关接口采用右下对齐，某些 SDPA causal 路径的定义不同。用2 queries、5 keys的显式 mask 对照，不能只测试方阵。详细例子见 [attention 契约](../../../../references/attention-backend-contracts.md)。

## 安装与 dispatch
flash-attn 通用包与 Hopper FA3 专项/其他后端不是同一能力集；仅安装 FA2 不能声称启用 FA3 FP8。Turing、Ampere、Ada、Hopper/Blackwell 的支持按目标版本查。记录选择到的 backend 和 fallback，不把 torch SDPA 调用名等同 FlashAttention 实际执行。

计时前生成并固定输入，做必要 warmup；分别报告 layout/contiguous copy、wrapper 与 kernel 时间。比较时相同 mask、精度、dropout、shape 和输出误差；随机数生成、动态分配或不同 fallback 不能混作算法收益。数值验收后再用 profiler 判断 memory/computation/launch 瓶颈。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `10-optimization/flash-attention/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [benchmarks](references/benchmarks.md)
- [transformers-integration](references/transformers-integration.md)

完整历史正文及本次具体勘误保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于追溯；运行规范使用本页与专题。
