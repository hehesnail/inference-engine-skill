# BigCode evaluation：生成与执行分离

代码评测包含生成候选与执行测试两阶段。数据、任务版本、prompt、模型/tokenizer、temperature/top-p、stop、max_tokens与样本数共同定义结果。

## 候选与分数
保存每个题目的所有原始候选，记录真正完成/截断/失败数量。pass@k估计要求每题候选数满足所用估计器条件（通常n≥k）；“必须200条”不是所有任务的通用规则。使用相同候选数据执行测试，避免重生成掩盖差异。

model output的截取、代码fence清理和语言解析会改变执行内容，保持原文与实际执行文本的映射。生成速度、测试耗时与模型质量分别统计。

## 执行环境
模型生成的代码在隔离的任务环境运行，限制权限、网络、资源和可访问路径；普通docker启动本身不证明隔离充分。沿当前工作区已有隔离流程执行，不自动下载/执行陌生依赖。拉取的镜像名与运行镜像须一致，保存digest和工具版本。

compare baseline/candidate使用同样依赖、timeout、测试数据和预处理。超时、编译失败、异常与错误答案分别报告，不能仅丢弃失败样本后算pass@k。见 [离线评测](../../../../references/offline-evaluation.md)。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `11-evaluation/bigcode-evaluation-harness/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [benchmarks](references/benchmarks.md)
- [custom-tasks](references/custom-tasks.md)
- [issues](references/issues.md)

完整历史正文及本次具体勘误保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于追溯；运行规范使用本页与专题。
