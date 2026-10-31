# Mamba 与混合 SSM 的推理接入

Mamba 的选择性状态空间递推把历史压缩到状态，纯 SSM 层不维护随序列增长的 attention KV。混合模型同时有 SSM/卷积状态与 attention KV；参数量、状态维度、层数和分片方式要从实际 config 读取。

## 从数学模块走到引擎
沿输入 projection、短卷积、selective scan/state update、gate 和 output projection 定位。prefill 的并行 scan 与 decode 的单步 update 需共用一致状态定义，比较同一 token 流的一次性与分块执行。Mamba/Mamba2 的 head/state 组织与 kernel 约束不同，不能套一份固定 d_state 或缓存公式。

接入 scheduler/runner 时检查状态池、slot 重排、prefix 状态快照、取消与复用、dummy/profile run、图捕获、prompt logprobs 和 speculative rollback。详细所有权与对照例子见 [循环状态与更新](../../../../references/stateful-inference.md)。

## 安装与测量
按目标 mamba-ssm、PyTorch、CUDA 和 causal-conv1d 的兼容组合选 wheel 或源码构建。`--no-build-isolation` 是复用当前构建依赖，并不代表只装二进制；shell 版本约束需要引号。已安装环境先定位实际加载文件，不因指南自动升级。

2.8B 参数按 FP16 仅原始权重约 5.6 GB（十进制），还需状态与 workspace；不是 28 GB。训练 gradient checkpointing 不解决推理引擎状态池的 OOM。固定内存与长上下文质量分开判断，删除无 workload 支持的“5×速度/50%显存”保证。

验证先用约定输入比较 logits/输出和状态续接，再测 prefill/decode、batch 与 context 分布。不要仅用一个层的 scan microbenchmark 推导整服务收益。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `01-model-architecture/mamba/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [architecture-details](references/architecture-details.md)
- [benchmarks](references/benchmarks.md)
- [training-guide](references/training-guide.md)

完整历史正文及本次具体勘误保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于追溯；运行规范使用本页与专题。
