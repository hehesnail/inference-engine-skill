# slime：rollout 与推理引擎的接线

本专题保留 Megatron/SGLang、data buffer、异步 rollout 与权重同步中和推理引擎直接相关的机制。训练奖励、数据制备和集群参数只有任务需要训练时才展开。

## 版本与状态
把 rollout sample、生成时 policy/adapter 版本、token IDs、采样与logprobs关联起来。异步buffer可混入旧策略数据，训练消费规则需明确；不能仅“提高sync间隔”就修复死锁或一致性问题。multi-turn工具交互保留各轮mask、stop与上下文边界。

权重同步沿训练导出→传输→各rank加载→draft/量化packed数据→cache失效→发布新版本检查。colocation/IPC避免某些网络copy，不免除同步和buffer存活期。配置训练8卡+rollout8卡不自动意味着两者共享8卡。

## 定位
SGLang crash先分加载、KV、workspace、CUDA与通信；提高mem_fraction_static可能挤压临时内存而更糟。actor梯度checkpointing不必解决rollout的KV OOM。
自定义generate/reward的导入路径、签名和样本schema从目标slime版本确认，旧指南函数不能当成固定公开接口。

实际状态、权重更新会话/部分失败和最小对照见 [循环状态与在线更新](../../../../references/stateful-inference.md)。完整训练示例保留在来源快照，执行前须固定框架版本和资源。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `06-post-training/slime/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [api-reference](references/api-reference.md)
- [troubleshooting](references/troubleshooting.md)

完整历史训练示例保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于查证版本和追溯。
