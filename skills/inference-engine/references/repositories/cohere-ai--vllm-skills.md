# cohere-ai/vllm-skills

固定提交：[2a9ed1ff5dd822a521a453f49eff84e8dd5a7071](https://github.com/cohere-ai/vllm-skills/tree/2a9ed1ff5dd822a521a453f49eff84e8dd5a7071)。

先读本地适配：[fork维护](../fork-maintenance.md)。

当前采用 5/5 项。采用项正文逐篇语义审阅；相关源码和附属资源的核对范围单列，不能推广成整仓每行已验证。技术修订与限制见 [语义核对](../semantic-audit.md)。

| 专项 | 用途 |
|---|---|
| [auto-rebase](../../library/cohere/skills/auto-rebase/GUIDE.md) | 发现upstream稳定release，保留fork并执行指定回归。 |
| [detect-upstream-base](../../library/cohere/skills/detect-upstream-base/GUIDE.md) | merge-base与最近可达tag作为定位线索，明确可回退SHA。 |
| [install-vllm](../../library/cohere/skills/install-vllm/GUIDE.md) | editable Python和预编译扩展分离、实际CLI与metadata校验，有助避免源码遮蔽误判。 |
| [local-test-runner](../../library/cohere/skills/local-test-runner/GUIDE.md) | 从Buildkite任务抽出对应pytest、cwd、环境、模型鉴权，适合已选CI失败复现。 |
| [rebase-assistant](../../library/cohere/skills/rebase-assistant/GUIDE.md) | 备份分支、保留fork意图、依赖差异归因、指定checks的rebase路线。 |
