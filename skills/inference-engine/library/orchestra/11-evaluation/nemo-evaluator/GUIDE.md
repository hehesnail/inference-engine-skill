# NeMo Evaluator：任务、endpoint与可追溯结果

先确认当前Evaluator版本、任务/harness适配器、模型接口与环境。评测容器、被测服务、代理与数据路径是不同层，容器内localhost不自动指宿主模型服务。

## 对齐输入输出
generation/chat任务与loglikelihood任务需要不同端点能力。一个chat completion成功不能证明所有任务可跑。固定模板、few-shot、sampling、max_tokens、tokenizer/模型revision、数据与输出转换；temperature0不保证跨硬件/backend绝对bitwise一致。

容器和服务部署参数来自当前环境；使用已有认证路径，密钥不放入 `docker login -p ...` 这类命令参数或日志。镜像tag与源码版本分别记录。

## 结果验收
记录样本原数、执行/失败/跳过、评分器版本和实际生成输出。质量与性能使用相同输入计划及计时边界；评测成功的含义是任务执行与结果完整，而不是只看进程启动或HTTP200。

大规模配置、集群launch和发布只在任务范围需要时使用；配套示例先对照所装版本。与 [离线评测](../../../../references/offline-evaluation.md) 联用。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `11-evaluation/nemo-evaluator/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [adapter-system](references/adapter-system.md)
- [configuration](references/configuration.md)
- [custom-benchmarks](references/custom-benchmarks.md)
- [execution-backends](references/execution-backends.md)

完整历史正文及本次具体勘误保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于追溯；运行规范使用本页与专题。
