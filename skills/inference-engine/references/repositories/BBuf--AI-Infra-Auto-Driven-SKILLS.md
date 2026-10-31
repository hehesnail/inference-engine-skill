# BBuf/AI-Infra-Auto-Driven-SKILLS

固定提交：[c592955044e896edd20890468c9bdbbe30ab0be6](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/c592955044e896edd20890468c9bdbbe30ab0be6)。

先读本地适配：[离线工具](../tools.md) · [模型接线](../model-integration.md)。

当前采用 11/12 项。采用项正文逐篇语义审阅；相关源码和附属资源的核对范围单列，不能推广成整仓每行已验证。技术修订与限制见 [语义核对](../semantic-audit.md)。

| 专项 | 用途 |
|---|---|
| [model-pr-optimization-history](../../library/bbuf/model-pr-optimization-history/GUIDE.md) | 按模型家族查询历史PR、区分生成快照与人工审阅，可为fast path风险与验证点提供候选。 |
| [llm-pipeline-analysis](../../library/bbuf/skills/llm-pipeline-analysis/GUIDE.md) | 由配置/源码/trace三者建立代表层流水线，分别列冷启动、稳态、prefill/decode、rank维度。 |
| [llm-serving-auto-benchmark](../../library/bbuf/skills/llm-serving-auto-benchmark/GUIDE.md) | 对齐工作负载/SLA与服务设置，手工运行候选并规范化结果，区分profiling与性能测量。 |
| [llm-serving-capacity-planner](../../library/bbuf/skills/llm-serving-capacity-planner/GUIDE.md) | 按启动日志重建每rank权重/KV/graph显存预算，用真实分配日志优先于理论payload。 |
| [llm-torch-profiler-analysis](../../library/bbuf/skills/llm-torch-profiler-analysis/GUIDE.md) | formal trace测时、eager trace补映射；区分sum/union/wall、PDL等待、实际backend与无法解析路径。 |
| [model-architecture-diagram](../../library/bbuf/skills/model-architecture-diagram/GUIDE.md) | 从已有索引按模型选原架构图，保留作者URL与图片来源，适合定位已有参考。 |
| [model-compute-simulation](../../library/bbuf/skills/model-compute-simulation/GUIDE.md) | 从模型config建立近似算子/FLOP预算，区分uniform TP/EP假设与真实测量，能定位待核对算子。 |
| [sglang-model-day0-support](../../library/bbuf/skills/model-optimization/sglang-model-day0-support/GUIDE.md) | Day0发布按配置/权重/registry/cache/协议/拓扑/质量性能建立显式证据矩阵，未知项与blocked lane透明。 |
| [sglang-humanize-review](../../library/bbuf/skills/sglang-humanize-review/GUIDE.md) | 从具体调用链、数据与历史maintainer讨论评估PR，强调正确性证据而非只看diff外观。 |
| [sglang-prod-incident-triage](../../library/bbuf/skills/sglang-prod-incident-triage/GUIDE.md) | 健康/队列/指标到trusted dump/replay再profile/bisect的渐进排查，保留原始事故输入与状态。 |
| [torch-profiler-layer-track](../../library/bbuf/skills/torch-profiler-layer-track/GUIDE.md) | 只添加/折叠展示轨道，原事件/时间戳/顺序不变；显式anchor偏移与unknown尾层，保留原trace可重建。 |

## 仅做范围筛选的入口

下列条目只核对名称与描述，未全文语义审阅，不认定无用；完整源码留在固定checkout，任务涉及其领域时再读。

| 来源入口 | 本轮范围 |
|---|---|
| [skills/sglang-explainer-video/SKILL.md](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/blob/c592955044e896edd20890468c9bdbbe30ab0be6/skills/sglang-explainer-video/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
