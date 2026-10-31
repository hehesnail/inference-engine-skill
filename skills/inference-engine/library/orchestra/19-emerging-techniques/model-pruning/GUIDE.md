# 模型剪枝与稀疏推理

剪枝需同时解决误差、稀疏表示与执行 kernel。把 dense 权重设为0通常不会减少 dense 文件大小或 GEMM FLOPs，也不保证加速。

## 方法契约
magnitude 用权重大小选择；Wanda 结合权重与输入 activation 的列统计，收集完整校准样本的 L2/二阶量而非不断覆盖为最后一批或取错维度。SparseGPT 使用相应层/校准统计与误差补偿，不能把伪代码类名当成通用库API。

N:M pattern、分组方向、对齐和硬件指令共同决定是否可压缩；2:4支持不意味着任意4:8都有同样加速。阈值相等的权重可能让 quantile mask 的实际稀疏率偏离目标，需要计数核对。微调时若未维护 mask，梯度可能令已剪权重重新生长。

## 引擎验证
保存 mask、稀疏格式、转换器与原权重身份；loader/dispatch必须真正执行 sparse kernel。比较校准集外的质量及相同workload内存、wrapper、kernel和整模型收益。模型输出/指标字典不能直接相减，先选择对应tensor或标量并对齐键。

结构化裁剪可能改变 head/hidden维度、RoPE、分片和checkpoint权重形状；从模型配置到runner与kernel一起接线。参见 [量化与稀疏契约](../../../../references/quantization-engine-contracts.md)。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `19-emerging-techniques/model-pruning/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [wanda](references/wanda.md)

完整历史正文及本次具体勘误保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于追溯；运行规范使用本页与专题。
