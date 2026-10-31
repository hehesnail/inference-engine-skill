# 推理优化候选、对照与保留

用于需要多轮候选的 kernel、runtime、serving 优化。吸收 KDA 的任务契约、Atrex 的精确源码绑定、KernelAgent 的测量隔离和 Hyperloom 的配置保留；只使用本次任务有价值的部分，不启动这些仓库的整套 agent 服务。

## 基线和候选的关系

区分原始基线（stock/original）、当前已验证最好版本（incumbent）与未验证候选（candidate）。原始基线回答总收益，incumbent 回答本次改动的增量收益。参考框架/理论上限可以帮助选择方向，但不能替换用户指定基线。

使用已有任务描述和验收口径确定工作负载、输入输出、约束及成功条件；缺失且会影响选择时才补问。单个小修复无需先写几份计划文件。持续优化则记录候选的父版本、改变的因素、预期机制和判定证据，方便恢复与排除重复实验。

## 能重放的最小记录

下面是字段示例，可放进既有报告、CSV 或 JSONL；无需创建新实验平台，也不要求把模型/trace 塞进 skill：

```json
{
  "candidate": "fused-rmsnorm-02",
  "parent": "fused-rmsnorm-01",
  "original_baseline": "baseline-record-id",
  "source": {"commit": "exact-tested-commit", "patch_digest": "if-uncommitted"},
  "hypothesis": "remove one exposed launch without changing inplace semantics",
  "workload": {"phase": "decode", "input_ids_artifact": "run/inputs.json", "shape_record": "run/shapes.json"},
  "runtime": {"image_digest": "record", "gpu": "actual-SKU", "effective_config": "run/effective-config.json"},
  "correctness": {"status": "passed", "evidence": "run/correctness.json"},
  "performance": {"status": "measured", "baseline_us": 100.0, "candidate_us": 93.0, "raw_samples": "run/times.csv"},
  "decision": "keep",
  "reason": "matched measurements satisfy the agreed criterion"
}
```

数值是格式示例，不是默认目标。实际配置包含最终生效的 flags/env、删除的 flags、unset 的 env、graph/compile/backend 模式及加载的 patch；只保存“新增的一个开关”不足以复现配置替换或恢复。记录 tokenizer/model revision、输入和 evaluator 身份到足以区分两次测量的程度。

证据须对应被测源码。测试后又编辑候选，旧证据只能覆盖仍相同的路径；不能把新文件借用旧结果直接标为通过。元数据变化可以凭源码树与条件相同复用证据，不机械重跑。

## 实验选择与测量

1. 从热点、实际 dispatch 和已有优化历史提出能区分原因的候选。尽可能分离 kernel 改动、服务参数与工作负载变化；必须联合调整时标明组合，别把整体收益归到一个组件。
2. 用最小但真实进入变更路径的正确性检查排除无效候选。原位数据、持久状态、随机性与 cache 初态要与基线对应。
3. 在同硬件、相同已授权资源和工作负载下做匹配性能对照。短时抖动影响决策时，可交错 A/B 或 ABBA，记录次序与样本；不为遵从某个框架固定重复次数。
4. 局部更快后检查是否影响真实热点或端到端目标。仅当瓶颈转移确实使原 operating point 不合适、且任务允许时再扫描并发/批量；不要重启全部调参矩阵。
5. 达到约定目标、可用候选耗尽或预算到达时收尾。授权 GPU 窗口内优先运行已准备好的有效对照，详细解析和写报告留后。

算法减少算量、精度/量化变化、缓存命中提高和实现本身更快是不同机制。保持它们的证据范围，不能混成一个 kernel speedup。

## 保留状态的含义

| 状态 | 含义 |
|---|---|
| keep | 约定的正确性和目标测量通过；保留精确代码与生效配置 |
| reject | 有有效证据证明回归、错误或达不到约定条件；保留原因 |
| inconclusive | 噪声/覆盖/trace 或输出缺失，无法做此次判断 |
| blocked | 当前候选缺依赖、资源或必要外部信息；可继续不依赖它的工作 |

未完成、不完整或无法解析的验收不能当成 passed。若本次约定无需质量评测，可以说明“未评测且不在本次范围”，不新增验收门槛。Hyperloom 上游某些 warm replay 路径允许缺质量分数时继续晋升；这里不继承把缺失证据当成已验证收益的做法。

保留最后一个经过验证的 incumbent，不因候选自报更快而覆盖它。回滚限于本次候选改动，保留用户其他修改及仍有用的失败证据。合并、发布或更换生产默认值由本次任务要求决定，keep 只表示实验结论。

完整agent工程机制见[工作流接线](agentic-kernel-workflows.md)；配置、量化和speculative候选见[具体优化场景](inference-optimization-cases.md)。
