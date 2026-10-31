# shen-shanshan/vllm-dev-skills

固定提交：[1ccafeebf71b5d922f351a0cd009a258b7474fac](https://github.com/shen-shanshan/vllm-dev-skills/tree/1ccafeebf71b5d922f351a0cd009a258b7474fac)。

先读本地适配：[fork与版本](../fork-maintenance.md) · [时间线分析](../host-device-analysis.md)。

当前采用 16/17 项。采用项正文逐篇语义审阅；相关源码和附属资源的核对范围单列，不能推广成整仓每行已验证。技术修订与限制见 [语义核对](../semantic-audit.md)。

| 专项 | 用途 |
|---|---|
| [infx-bench-script-generator](../../library/vllm-dev/skills/infx-bench-script-generator/GUIDE.md) | 将单机InferenceX recipe深合并后生成容器/服务/客户端三脚本；保留真实与golden acceptance的区别。 |
| [vllm-benchmark-result-analysis](../../library/vllm-dev/skills/vllm-benchmark-result-analysis/GUIDE.md) | 把before/after标准文本解析为百分比表，适用已经确认可比条件的描述性汇总。 |
| [vllm-dev-task-discovery](../../library/vllm-dev/skills/vllm-dev-task-discovery/GUIDE.md) | 从issue/merged PR/TODO/roadmap提炼带来源的可行动开发机会。 |
| [vllm-feature-design](../../library/vllm-dev/skills/vllm-feature-design/GUIDE.md) | 以数据结构、critical path和共享状态约束设计最小实现，适用新feature方案。 |
| [vllm-feature-tutorial](../../library/vllm-dev/skills/vllm-feature-tutorial/GUIDE.md) | 从实际类、接口和控制流做功能讲解，保留version/date和源文件索引。 |
| [vllm-model-tutorial](../../library/vllm-dev/skills/vllm-model-tutorial/GUIDE.md) | 模型论文/config/processor/forward/权重路径联合分析，模型适配前可用。 |
| [vllm-multimodal-open-issue-analyzer](../../library/vllm-dev/skills/vllm-multimodal-open-issue-analyzer/GUIDE.md) | 按关键词批量发现多模态issue并多标签分类，适合作为候选清单。 |
| [vllm-pr-desc-generator](../../library/vllm-dev/skills/vllm-pr-desc-generator/GUIDE.md) | 依据diff、讨论与真实验证材料起草PR描述，不捏造性能数字；可复用上游Claims/Validation结构。 |
| [vllm-pr-summary](../../library/vllm-dev/skills/vllm-pr-summary/GUIDE.md) | 把PR diff/评论/动机与风险整理为可读报告，适合实现学习和定向review。 |
| [vllm-rfc-generator](../../library/vllm-dev/skills/vllm-rfc-generator/GUIDE.md) | RFC动机、设计、兼容与依赖PR分类，适合用户明确要方案文稿。 |
| [vllm-rocm-code-dev](../../library/vllm-dev/skills/vllm-rocm-code-dev/GUIDE.md) | ROCm能力门控、base hook默认兼容、fallback与其他backend影响分析，适合ROCm专项改动。 |
| [vllm-rocm-env-prepare](../../library/vllm-dev/skills/vllm-rocm-env-prepare/GUIDE.md) | 复用容器预编译so并对齐Python源码的开发环境思路，可用于指定镜像复现。 |
| [vllm-rocm-pr-review](../../library/vllm-dev/skills/vllm-rocm-pr-review/GUIDE.md) | 具备具体触发输入gate、跨完整函数/调用方验证、性能数字溯源和CI基线区分，适合ROCm review。 |
| [vllm-technical-blog-writer](../../library/vllm-dev/skills/vllm-technical-blog-writer/GUIDE.md) | 以单个问题与连贯输入例子解释真实调用链，既有草稿保留用户内容；图必须落盘渲染。 |
| [vllm-test-generator](../../library/vllm-dev/skills/vllm-test-generator/GUIDE.md) | 沿邻近fixture、model/API层与Buildkite任务选择测试，适合目标feature回归。 |
| [vllm-vs-atom-decode-trace-comparison](../../library/vllm-dev/skills/vllm-vs-atom-decode-trace-comparison/GUIDE.md) | 以同模型同配置trace做逻辑算子级比较，保留融合与多流区别及无法比较性能的边界。 |

## 仅做范围筛选的入口

下列条目只核对名称与描述，未全文语义审阅，不认定无用；完整源码留在固定checkout，任务涉及其领域时再读。

| 来源入口 | 本轮范围 |
|---|---|
| [skills/skill-creator/SKILL.md](https://github.com/shen-shanshan/vllm-dev-skills/blob/1ccafeebf71b5d922f351a0cd009a258b7474fac/skills/skill-creator/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
