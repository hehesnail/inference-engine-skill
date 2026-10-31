# lm-evaluation-harness：可比较的模型评测

固定任务定义、数据revision/分割、prompt/template、tokenizer、few-shot样本选择、采样参数与模型权重。harness版本、task YAML与实际metric/filter键都是结果身份的一部分。

## 任务与接口
区分 loglikelihood、rolling loglikelihood 与 generation。OpenAI-compatible chat端点不自动支持所有评分任务；模板/系统提示改变也会改变比较对象。固定few-shot数的历史例子不代替任务当前默认值或用户约定。

HF加载路径要是真正兼容的模型目录；Lightning训练checkpoint不自动等同HF pretrained目录。vLLM/SGLang backend的可选参数需按当前harness与引擎版本确认。记录adapter、量化、长度截断和batch条件。

## 运行与解释
保存逐样本输出、实际执行数、被筛选/失败数及metric keys。结果字典可能含filter后缀，不能只猜“acc”键。降低样本数、换few-shot或改变模板会改变评测含义，不能包装为语义不变的加速。

并行/GPU数匹配后再谈引擎速度；HF单卡与vLLM两卡只能说明不同资源配置，不能直接推出backend收益。质量分数与throughput分别报告，重复测量的随机性与置信信息按任务需要提供。见 [离线评测](../../../../references/offline-evaluation.md)、[Tokenizer契约](../../../../references/tokenizer-and-structured-output.md)。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `11-evaluation/lm-evaluation-harness/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [api-evaluation](references/api-evaluation.md)
- [benchmark-guide](references/benchmark-guide.md)
- [custom-tasks](references/custom-tasks.md)
- [distributed-eval](references/distributed-eval.md)

完整历史正文及本次具体勘误保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于追溯；运行规范使用本页与专题。
