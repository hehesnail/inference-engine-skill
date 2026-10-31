# Instructor：结构化提取、验证与重试

Instructor 包装provider调用并结合schema验证；它不自动等于引擎token级grammar控制。明确所用provider、mode、模型与响应格式能力。

## Schema与错误
Pydantic字段类型、业务validator、缺失/null规则分别设计。用默认年龄/日期掩盖无法提取的数据会把结构成功变成事实错误。日期正则允许不存在日期时应再解析；约束文本是否完整匹配也需明确。

streaming Partial 对象字段可能仍为None，先判可用再切片/消费；最终对象另作完整校验。异常可能经过wrapper封装，重试耗尽不必是裸ValidationError。provider模式不通用，例如不能机械把OpenAI的Mode.JSON套给Anthropic。

## 服务与评测
重试增加实际请求数、tokens和端到端延迟，记录第一次成功率、最终成功率、每次失败类别及预算。max_tokens截断/取消/网络失败与schema业务失败分开。并发时请求状态和校验错误不得串到别的请求。

更深的grammar、stop、tokenizer与stream边界见 [结构输出契约](../../../../references/tokenizer-and-structured-output.md)。使用现有wrapper流程，不为普通推理排障另造agent框架。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `16-prompt-engineering/instructor/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [examples](references/examples.md)
- [providers](references/providers.md)
- [validation](references/validation.md)

完整历史正文及本次具体勘误保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于追溯；运行规范使用本页与专题。
