# AMD-AGI/Hyperloom

固定提交：[a646a9c14c827800be4df8cd862da89100b5b3df](https://github.com/AMD-AGI/Hyperloom/tree/a646a9c14c827800be4df8cd862da89100b5b3df)。

先读本地适配：[候选、stack与resume](../agentic-kernel-workflows.md)。

当前采用 17/17 项。采用项正文逐篇语义审阅；相关源码和附属资源的核对范围单列，不能推广成整仓每行已验证。技术修订与限制见 [语义核对](../semantic-audit.md)。

| 专项 | 用途 |
|---|---|
| [analyze-upstream-pr](../../library/hyperloom/.claude/skills/analyze-upstream-pr/GUIDE.md) | 把PR机制、目标dispatch、是否已包含、操作点差异分开，未知不当缺失；用于上游patch评估。 |
| [check-prior-art](../../library/hyperloom/.claude/skills/check-prior-art/GUIDE.md) | 关键词+路径重叠+diff语义+git inclusion交叉检查，报告截断与未搜范围。 |
| [review-pr](../../library/hyperloom/.claude/skills/review-pr/GUIDE.md) | 沿producer/consumer追契约，在merge-base复现并反证 findings，适用于引擎review。 |
| [hyperloom-remote-mn-qwen3-30b](../../library/hyperloom/docs/how-to/multi-node/hyperloom-remote-mn-qwen3-30b/GUIDE.md) | PD角色TP/EP、共享路径、服务端和客户端env、同session恢复。 |
| [hyperloom-custom-advanced](../../library/hyperloom/examples/hyperloom-custom-advanced/GUIDE.md) | 显式workload参数、session独立元数据、进程与wrapper PID区分。 |
| [hyperloom-qwen3-14b-fp8-12h-forge](../../library/hyperloom/examples/hyperloom-qwen3-14b-fp8-12h-forge/GUIDE.md) | 可控单变量forge/geak对比，检查记录backend而非相信shell export。 |
| [hyperloom-qwen3-14b-fp8-12h](../../library/hyperloom/examples/hyperloom-qwen3-14b-fp8-12h/GUIDE.md) | ATOM/其他engine Python和plugin隔离，模型/启动配置与状态一致。 |
| [hyperloom-qwen3-8b-3h](../../library/hyperloom/examples/hyperloom-qwen3-8b-3h/GUIDE.md) | 框架配置与源码候选分支，禁kernel时预算重新分配是该运行时特性。 |
| [hyperloom-workload-optimizer](../../library/hyperloom/examples/skills/hyperloom-workload-optimizer/GUIDE.md) | 区分bootstrap/setup/run所有者，采集实际session状态。 |
| [critic](../../library/hyperloom/src/hyperloom/agents/critic/GUIDE.md) | 区分产生证据的探索、功能enablement、性能晋升，避免用尚未产生的benchmark阻断探索。 |
| [quantization](../../library/hyperloom/src/hyperloom/agents/quantization/GUIDE.md) | 隔离产物、明确失败分类、可执行修复假设才retry、源/量化模型对照。 |
| [inference_optimizer](../../library/hyperloom/src/hyperloom/inference_optimizer/GUIDE.md) | baseline/current_best/optimization_stack、持久候选去重、源与二进制恢复、重启加载、有限真实spawn预算。 |
| [breakdown](../../library/hyperloom/src/hyperloom/inference_optimizer/breakdown/GUIDE.md) | 确定性breakdown区分measured/projected/null、不在action中间读不一致状态。 |
| [multi_node](../../library/hyperloom/src/hyperloom/inference_optimizer/multi_node/GUIDE.md) | 多节点endpoint、所有rank重启、真实completion readiness和候选abort证据。 |
| [hyperloom-setup](../../library/hyperloom/src/hyperloom/skills/hyperloom-setup/GUIDE.md) | 使用现有用户选项、实际Python import/CLI验证、框架plugin隔离。 |
| [hyperloom-kb](../../library/hyperloom/src/hyperloom_kb/skills/hyperloom-kb/GUIDE.md) | 历史经验有schema、标签快照和排除理由，可回溯恢复。 |
| [kernel-trace-analysis](../../library/hyperloom/src/kernelforge/knowledge/local_wiki/resources/languages/flydsl/skills/profile/kernel-trace-analysis/GUIDE.md) | ATT定位指令stall、PMC补缓存/带宽、寄存器/LDS限制拆开解释。 |
