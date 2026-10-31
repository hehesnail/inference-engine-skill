# RWKV 推理与循环状态

RWKV 以递归状态进行推理，不同世代的 time-mix/channel-mix、状态布局、精度和运行策略不同。固定大小状态节省逐 token KV，但不保证任意长历史的召回质量。

## 正确驱动递归 API
环境变量若影响后端选择，要在导入相应库之前设置。tokenizer 和模型 checkpoint 配对；首次将完整 prompt 编码成 IDs 后 forward，后续将新采样的 token（或 token 列表）和返回 state 再送入。`forward(None, state)` 不能代替输入 token；逐字符 encode 拼接也不等同整段 encode。

解码时调用 tokenizer 所支持的 token 序列接口，保留 UTF-8 尚未完成的字节片段。stop token、文本 stop 与输出显示边界分开；不要在控制台逐 token 打印的测量里混入 I/O 后再称 kernel 性能。

## 接入 serving
请求拥有独立 state；批处理重排必须同步 state slot。续写保留状态，新请求/slot 复用要重置。分支、prefix 共享与 speculative 解码需要状态复制或快照回滚。state tuning/LoRA 更新需区分参数更新与请求状态更新，不把旧 state 接到不同版本的权重。

内存按实际张量形状、层数、dtype 与 batch 计算；不同 RWKV 版本不可共用固定每层 state 公式。检查策略字符串实际解析到的 device/dtype/量化，而不是从名称猜测。
 
[状态生命周期](../../../../references/stateful-inference.md) 给出分块、污染和在线更新对照。上游训练示意不自动成为可运行 trainer API；训练只在当前任务需要时查目标版本实现。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `01-model-architecture/rwkv/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [architecture-details](references/architecture-details.md)
- [rwkv7](references/rwkv7.md)
- [state-management](references/state-management.md)

完整历史正文及本次具体勘误保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于追溯；运行规范使用本页与专题。
