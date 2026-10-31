# miles：低精度 rollout 与训练/推理对齐

miles相关资料中有用的是 FP8/INT4 流程、routing replay、speculative rollout、权重同步和样本版本，不是未经测量的固定加速承诺。

## 对齐边界
MoE Rollout Routing Replay 保持专家选择只能消除部分差异；不同kernel、量化scales、累加顺序、mask或logprob定义仍会不同，不能宣称bitwise logits一致或KL必为零。FP8 QAT导出的scales、权重和模型config需被推理backend真正消费。

speculative draft随训练漂移时记录target/draft版本、接受率与最终质量，partial rollout记录每段实际版本。提高更新频率不自动解决版本不一致；先核对会话、在途请求和cache身份。

## 接线与验证
IPC/colocation要求权重buffer的生命周期、rank协调与失败恢复；更新成功HTTP不代表所有组件一致。固定tokens和采样变换，比较训练/rollout logprobs及必要中间tensor，再解释优化收益。任务不涉及训练时不加载完整Megatron/RL环境。

见 [状态与在线更新](../../../../references/stateful-inference.md)、[量化契约](../../../../references/quantization-engine-contracts.md)、[推测解码](../../../../references/speculative-decoding-contracts.md)。版本专属CLI和“零KL/固定倍速”不作为本工作台要求。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `06-post-training/miles/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [api-reference](references/api-reference.md)
- [troubleshooting](references/troubleshooting.md)

完整历史训练示例保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于查证版本和追溯。
