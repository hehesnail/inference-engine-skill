# Guidance：约束与程序控制的边界

Guidance 可组合文本、选择、重复和生成表达式；实际控制能力依模型 backend。程序拼接方便不代表远程provider暴露任意logits或token healing。

## 接口与语义
按目标版本使用 gen、select、regex等真实API。所核对 gen 签名没有通用 grammar 参数，不能将 XML-like 字符串 `grammar="<name>..."` 当成 parser。远程 OpenAI/Anthropic backend与本地模型能力单独核对，受provider支持的响应格式和采样接口限制。

grammar/regex管语法，Pydantic或业务校验管语义，提示词只提供倾向。日期正则、邮箱格式与事实正确不是一回事。多轮程序循环要有终止、token与任务边界；遵循当前授权，不自动启用无限代理运行。

## 工具与测量
模型输出的工具名和参数经明确映射与校验调用，不对字符串直接 eval。只有本任务授权的动作才执行。记录一次业务请求引发的所有生成、retry、token和延迟。

模型、tokenizer、模板、constraint backend与stop的组合见 [Tokenizer与结构输出契约](../../../../references/tokenizer-and-structured-output.md)。旧配套示例作为版本定位材料，不能跨backend复制保证。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `16-prompt-engineering/guidance/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [backends](references/backends.md)
- [constraints](references/constraints.md)
- [examples](references/examples.md)

完整历史正文及本次具体勘误保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于追溯；运行规范使用本页与专题。
