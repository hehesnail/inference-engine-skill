# cfregly/gpu-perf-tune

固定提交：[25d74d031732e2f76d30a43e44c3434b588f06b3](https://github.com/cfregly/gpu-perf-tune/tree/25d74d031732e2f76d30a43e44c3434b588f06b3)。

先读本地适配：[workload与优化](../inference-optimization-cases.md)。

当前采用 32/33 项。采用项正文逐篇语义审阅；相关源码和附属资源的核对范围单列，不能推广成整仓每行已验证。技术修订与限制见 [语义核对](../semantic-audit.md)。

| 专项 | 用途 |
|---|---|
| [analyze-zymtrace-workload](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/analyze-zymtrace-workload/GUIDE.md) | 匹配CPU/GPU窗口、进程和过滤器，保留样本占比分母与原始查询。 |
| [evidence-bundle-init](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/evidence-bundle-init/GUIDE.md) | run-id及命令/stdout/stderr/exit四件套，源码/镜像/overlay身份可迁移。 |
| [inference-aa-workload](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/inference-aa-workload/GUIDE.md) | 区分TTFT和首内容TTFO，固定请求shape、模型tokenizer和reasoning字段。 |
| [inference-capacity-sizing](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/inference-capacity-sizing/GUIDE.md) | 以SLA过滤并报告容量/成本与利用率假设，适合规划。 |
| [inference-dcgm-correlate](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/inference-dcgm-correlate/GUIDE.md) | 把遥测窗口与实际bench、node和设备身份对齐；保留冻结查询。 |
| [inference-decode-step-budget](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/inference-decode-step-budget/GUIDE.md) | 从每decode步时序分解host/device/comm，核对外部TPOT与trace覆盖。 |
| [inference-fleet-leaderboard](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/inference-fleet-leaderboard/GUIDE.md) | 跨campaign Pareto与质量条件分开，保留每点shape和成本。 |
| [inference-graph-diff](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/inference-graph-diff/GUIDE.md) | 确认编译选择真正生效、保留图和lowered-code差异。 |
| [inference-kernel-ncu-profile](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/inference-kernel-ncu-profile/GUIDE.md) | 限定kernel与replay范围、区分忙碌率/指令/byte计数、保留真实生产baseline。 |
| [inference-kernel-profile](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/inference-kernel-profile/GUIDE.md) | timeline/NVTX/图节点与绝对duration补充采样，保留工具版本。 |
| [inference-kernel-whitebox-debug](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/inference-kernel-whitebox-debug/GUIDE.md) | 同输入单tile operands/accumulator/epilogue追踪及精确实例化standalone对照。 |
| [inference-known-good-config](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/inference-known-good-config/GUIDE.md) | 将必需启动flag、版本范围、证据与完整champion配置分开存。 |
| [inference-model-eval](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/inference-model-eval/GUIDE.md) | 模型质量与性能分别验收，记录同endpoint和基线。 |
| [inference-model-optimize](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/inference-model-optimize/GUIDE.md) | 先身份/准备/基线，再最小profile、限定矩阵、质量、best保留，集成价值高。 |
| [inference-perf-baseline-bridge](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/inference-perf-baseline-bridge/GUIDE.md) | 混合指标按方向和单位比较，null显示未测而非零，保留raw数据。 |
| [inference-perf-bench](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/inference-perf-bench/GUIDE.md) | 固定请求协议、客户端位置、源码交付、基线与失败数。 |
| [inference-perf-synthesize](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/inference-perf-synthesize/GUIDE.md) | 跨trace/遥测/源码解释矛盾，rank lever与测量证据绑定，再同条件A/B。 |
| [inference-perf-tune-report](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/inference-perf-tune-report/GUIDE.md) | 结构化atlas、失败/partial显式显示、保留来源和渲染完整性。 |
| [inference-performance-hints](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/inference-performance-hints/GUIDE.md) | lower bound重叠取max、串行相加；按频率及测量份额排假设、预设停止条件。 |
| [inference-quantize-calibrate](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/inference-quantize-calibrate/GUIDE.md) | 校准语料、量化覆盖、导出加载和质量/perf分阶段，vendor/base/ours角色分明。 |
| [inference-spec-decode-service](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/inference-spec-decode-service/GUIDE.md) | traffic/corpus/tokenizer/命令身份与独立accepted token和延迟gate。 |
| [inference-spec-decode-train](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/inference-spec-decode-train/GUIDE.md) | capture/train/export/serve契约分离，训练acc不等servingacceptance。 |
| [inference-spec-decode-tune](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/inference-spec-decode-tune/GUIDE.md) | 限定训练候选成本、同consumed samples筛代理、最终serving选胜者。 |
| [inference-tune-sweep](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/inference-tune-sweep/GUIDE.md) | 小矩阵由机制驱动，实际调度receipt，同图/缓存、近噪声保留baseline。 |
| [inference-value-ledger](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/inference-value-ledger/GUIDE.md) | findings、源码、campaign和生命周期显式join，记录失败避免重试。 |
| [inference-workload-profile](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/inference-workload-profile/GUIDE.md) | 请求分布、内容mix和corpus来源帮助draft贴近traffic。 |
| [k8s-troubleshooting](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/k8s-troubleshooting/GUIDE.md) | 先时间线和previous日志，再查依赖/事件/资源，按假设最小只读检查。 |
| [mirage-graph-coverage](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/mirage-graph-coverage/GUIDE.md) | declared-vs-consumed图接线审计补单算子测试，适合新模型/融合路径。 |
| [perf-baseline-diff](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/perf-baseline-diff/GUIDE.md) | 方向明确不可混latency和throughput，来源快照及差值可追溯。 |
| [perf-baseline-record](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/perf-baseline-record/GUIDE.md) | 不可覆盖的baseline版本、原始payload、来源和可选schema，适合回归。 |
| [prometheus-anchored-query](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/prometheus-anchored-query/GUIDE.md) | 窄scope/time/datasource/labels、保存原始PromQL响应。 |
| [zymtrace-anchored-query](../../library/gpu-perf-tune/plugins/profile-and-optimize/skills/zymtrace-anchored-query/GUIDE.md) | 先schema、pod/时间边界、原始SQL和响应，CPU/GPU同窗join。 |

## 仅做范围筛选的入口

下列条目只核对名称与描述，未全文语义审阅，不认定无用；完整源码留在固定checkout，任务涉及其领域时再读。

| 来源入口 | 本轮范围 |
|---|---|
| [plugins/profile-and-optimize/templates/skill/SKILL.md](https://github.com/cfregly/gpu-perf-tune/blob/25d74d031732e2f76d30a43e44c3434b588f06b3/plugins/profile-and-optimize/templates/skill/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
