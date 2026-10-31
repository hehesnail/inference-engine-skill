# 长上下文：位置、attention、缓存与质量

长上下文方案改变 positional encoding、attention/window、训练分布或检索结构；增加 max_model_len 不能独自证明模型会正确利用更长历史。

## RoPE 与 ALiBi
RoPE 的频率、缩放方案、维度布局（split-half/interleaved）与绝对 position 要匹配模型。chunk prefill 或 prefix 命中后位置从已处理长度延续，不能每块从0重启；旋转前后布局和缓存中 K 是否已旋转一致。
在 CUDA tensor 上构造频率时明确 device；`type_as(cpu_tensor)` 会同时改变 dtype/device，不能将其误作只转换 dtype。

ALiBi 给注意力分数加位置偏置，不替代 causal mask，也不保证无限长质量；构造完整 S×S bias 可能失去原有节省显存的收益。看 [attention 位置与mask契约](../../../../references/attention-backend-contracts.md)。

## 加载与容量
配置应在模型构建/权重加载的适当阶段生效；已实例化模块可能缓存 RoPE 参数，事后改 config 字段不保证更新。`from_config` 产生随机初始化模型，不是原权重的长上下文版本。基础模型长度从具体 config 读取，例如不能将所有 Llama2 基线写成2048。

KV、SSM状态、workspace、图池分别估算，采用引擎实际分片与 cache 日志。反复重复一段字符串不等于严格指定 token 数。prefix/滑窗/分块改变有效 work，性能对照注明实际 token IDs 与被保留上下文。

## 质量验证
按任务选择长距离检索、位置敏感性或真实长文任务；needle 命中不能覆盖所有能力。相同模板/tokenizer、prefill分块与窗口规则下对比数值和任务质量，再测 prefill、decode、峰值内存与端到端。微调/位置插值/YaRN等候选只在任务要求时展开，不能把历史论文倍率当成本服务保证。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `19-emerging-techniques/long-context/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [extension_methods](references/extension_methods.md)
- [fine_tuning](references/fine_tuning.md)
- [rope](references/rope.md)

完整历史正文及本次具体勘误保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于追溯；运行规范使用本页与专题。
