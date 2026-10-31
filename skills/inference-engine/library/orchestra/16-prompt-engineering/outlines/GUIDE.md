# Outlines：模型适配、约束编译与生成状态

Outlines 将类型/schema/regex/grammar约束连接到生成模型。版本升级改变了adapter与Generator接口；旧 generate.json 与新API不能混装。先查看所安装版本公开入口和所用模型backend。

## 约束层
有限正则与一般CFG不是同一计算模型；递归grammar不能统称FSM。tokenizer映射、schema支持子集、编译与cache影响能力和开销。Pydantic任意Python validator并不自动编译成token级约束。

零运行开销与100%有效输出都不成立为通用承诺。冷编译、逐步mask、dead end、EOS、max_tokens、stop、取消与provider错误需单独定义。完整JSON语法不证明日期、数值关系、引用或事实正确。

## 引擎接线
固定revision/模板，核对每个请求的grammar state、batch重排、prefix缓存与speculative回滚。复用compiled grammar不等于复用可变解析位置。用合法、非法、截断与并发输入验证实际路径，按当前任务选择必要边界。

读 [Tokenizer与结构输出契约](../../../../references/tokenizer-and-structured-output.md) 获取跨backend分层；配套backend/examples保留历史实现线索，执行前以目标API为准。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `16-prompt-engineering/outlines/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [backends](references/backends.md)
- [examples](references/examples.md)
- [json_generation](references/json_generation.md)

完整历史正文及本次具体勘误保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于追溯；运行规范使用本页与专题。
