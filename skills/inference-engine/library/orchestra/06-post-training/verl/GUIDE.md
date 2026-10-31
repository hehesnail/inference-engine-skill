# verl：HybridFlow 与 rollout 接口

区分训练actor/reference/critic、rollout引擎与控制面。资源和checkpoint如何分片/重组、推理请求怎样生成、logprobs怎样交给训练，是引擎接入的核心。

## 接线
固定rollout backend和版本、模型/tokenizer、chat template、stop、多轮loss mask、adapter与policy版本。LoRA只同步增量与同步merged weights有不同cache/packed weight处理。async或partial rollout需定义版本滞后和logprob重算，不能只开启一个flag便假定一致。

在训练与推理共用卡时，明确暂停/唤醒、权重offload、KV/SSM/graph池和内存所有权。OOM按actor、rollout、reference或logprob阶段定位；降低actor训练microbatch/gradient checkpoint不一定影响rollout KV。

## 使用范围
配置键、FSDP/Megatron后端与依赖兼容按当前verl版本读取，不沿用旧文“vLLM>=某版本且避开所有0.7.x”的永久规则。shell版本约束带引号；八张GPU是某个训练例子，不是框架通用最低需求。

把实际权重更新与固定输入验证落到 [状态专题](../../../../references/stateful-inference.md)，再按需要扩展训练任务。普通推理排障无需自动启动PPO/GRPO训练或集群。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `06-post-training/verl/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [api-reference](references/api-reference.md)
- [troubleshooting](references/troubleshooting.md)

完整历史训练示例保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于查证版本和追溯。
