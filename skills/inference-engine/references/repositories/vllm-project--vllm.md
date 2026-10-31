# vllm-project/vllm

固定提交：[7b77ed581d92fa14e34ab43344837bb1ca1b4257](https://github.com/vllm-project/vllm/tree/7b77ed581d92fa14e34ab43344837bb1ca1b4257)。

先读本地适配：[原15项复核](../upstream-vllm-audit.md)。

当前采用 9/9 项。采用项正文逐篇语义审阅；相关源码和附属资源的核对范围单列，不能推广成整仓每行已验证。技术修订与限制见 [语义核对](../semantic-audit.md)。

| 专项 | 用途 |
|---|---|
| [check-api-compat](../../library/vllm/.agents/skills/check-api-compat/GUIDE.md) | API协议排障：确定surface/endpoint/field后，比对模型声明、消费者、流终止与omitted/null；能指导服务协议任务。 |
| [ci-fails-buildkite](../../library/vllm/.agents/skills/ci-fails-buildkite/GUIDE.md) | 已完成的Buildkite失败日志定位与清洗；明确缓存文件及强制刷新，避免把历史日志当当前结果。 |
| [debug-ima](../../library/vllm/.agents/skills/debug-ima/GUIDE.md) | 把异步IMA收敛到首个异常、实际binary与cuda core；本地参考补齐了纯外链指南的离线操作能力。 |
| [kernel-microbenchmark](../../library/vllm/.agents/skills/kernel-microbenchmark/GUIDE.md) | 区分单算子时延、生产wrapper端到端、跨rank MAX与冷热缓存；适用kernel候选的公平比较。 |
| [model-microbenchmark](../../library/vllm/.agents/skills/model-microbenchmark/GUIDE.md) | MRV2真实execute_model的前向隔离计时；语法明确KV含query、prefix-hit与chunk、spec verify以及drafter边界。 |
| [model-recipes](../../library/vllm/.agents/skills/model-recipes/GUIDE.md) | 按HF checkpoint、variant、硬件、拓扑查配方；本地YAML与网站已解析command分开，正确区分recipe验证与具体命令验证。 |
| [pr-checklist](../../library/vllm/.agents/skills/pr-checklist/GUIDE.md) | 高价值是沿共享执行路径核对契约、测试真正进入新分支、PD请求流与可复现validation；可指导引擎PR review。 |
| [transformers-upgrade](../../library/vllm/.agents/skills/transformers-upgrade/GUIDE.md) | 配置与meta模型双dump捕捉字段默认值、shape、MTP text_config和remote-code接管；适用Transformers升级、去vendoring。 |
| [triton-kernel-writing](../../library/vllm/.agents/skills/triton-kernel-writing/GUIDE.md) | 聚焦Triton与Python语义差异、提前int64转换、grid维度、offset-view触发新变体，能指导KV索引和decode/prefill kernel实现。 |
