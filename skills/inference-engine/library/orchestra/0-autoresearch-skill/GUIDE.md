# 面向推理引擎的迭代研究

保留这套工程有价值的循环：提出可证伪假设、建立对照、一次改动、保存原始结果、分析失败、选择下一项有信息量的实验。研究的对象与结束条件来自当前任务。

## 最小可追溯记录
每项实验记录假设、基线/候选身份、输入与状态初态、命令、环境、实际执行/失败数、结果和解释。预期收益与实际证据分开，失败、无收益和异常同样保存；下一轮要能追溯为何选择这个方向。

基线固定且质量通过后再比较速度。对状态ful kernel还需恢复KV/循环状态，图replay需变化输入验证。多轮实验按 [优化工作流](../../../references/optimization-workflow.md) 与 [数值验收](../../../references/numerical-validation.md) 执行。

## 任务边界
研究循环不要求固定20分钟cron、无限运行、自动使用任何GPU、发布论文或给他人发消息。是否新开后台任务、花费资源、提交/发布由当前用户要求决定。git preregistration、论文式报告或多agent只是适用场景的选择，不是普通bugfix的先决条件。

窗口内优先完成有价值的已准备GPU实验；深入分析可以在后续CPU阶段进行。获得足够证据或约定实验完成即收束，不为维持循环凑无效测试。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `0-autoresearch-skill/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [agent-continuity](references/agent-continuity.md)
- [progress-reporting](references/progress-reporting.md)
- [skill-routing](references/skill-routing.md)

完整历史正文及本次具体勘误保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于追溯；运行规范使用本页与专题。
