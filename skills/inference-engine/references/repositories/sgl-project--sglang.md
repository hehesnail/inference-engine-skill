# sgl-project/sglang

固定提交：[8368f86fd7a71b07e2b7dda5d4146695ec54cbfa](https://github.com/sgl-project/sglang/tree/8368f86fd7a71b07e2b7dda5d4146695ec54cbfa)。

先读本地适配：[新专项接线](../sglang-specialized-paths.md)。

当前采用 33/33 项。采用项正文逐篇语义审阅；相关源码和附属资源的核对范围单列，不能推广成整仓每行已验证。技术修订与限制见 [语义核对](../semantic-audit.md)。

| 专项 | 用途 |
|---|---|
| [add-jit-kernel](../../library/sglang/.agents/skills/add-jit-kernel/GUIDE.md) | 完整JIT内核从CUDA/TVM FFI验证、编译key、lazy registry到tests/benchmark，可作为轻量算子新增路由。 |
| [add-sgl-kernel](../../library/sglang/.agents/skills/add-sgl-kernel/GUIDE.md) | AOT从launcher、schema、CMake、Python wheel导出到runtime lazy wrapper、正确性/benchmark接线。 |
| [babysit-pr-to-pass-ci](../../library/sglang/.agents/skills/babysit-pr-to-pass-ci/GUIDE.md) | 按指定workflow和最新PR head跟踪、分类根失败/级联/infra，限定重跑与发布范围。 |
| [ci-test-audit](../../library/sglang/.agents/skills/ci-test-audit/GUIDE.md) | 具体文件+证据置信度审计测试成本、真实路径/独立oracle、状态隔离、注册与环境。 |
| [ci-workflow-guide](../../library/sglang/.agents/skills/ci-workflow-guide/GUIDE.md) | 说明注册AST、分片、gate、current-head重跑与级联失败，帮助定位真正CI根因。 |
| [clean-startup-log](../../library/sglang/.agents/skills/clean-startup-log/GUIDE.md) | 保存启动全量日志、区分有用告警与噪音、追溯emitter，避免无关日志抑制。 |
| [compute-mamba-ratio](../../library/sglang/.agents/skills/compute-mamba-ratio/GUIDE.md) | 解释hybrid state/KV双池和近似平衡，区分cache策略、overlap、spec、DCP、dtype。 |
| [cookbook-add-model](../../library/sglang/.agents/skills/cookbook-add-model/GUIDE.md) | 配置驱动cookbook仅录实测cells和匹配模型事实，数据与共享render分离。 |
| [cookbook-migrate-model](../../library/sglang/.agents/skills/cookbook-migrate-model/GUIDE.md) | 旧页面迁移保留原命令/测量出处，逐combo独立token等价检查与导航锚点。 |
| [cookbook-review-pr](../../library/sglang/.agents/skills/cookbook-review-pr/GUIDE.md) | cookbook review关注schema、真实model/parserid、测量harness、命令展示与数据同源。 |
| [debug-cuda-crash](../../library/sglang/.agents/skills/debug-cuda-crash/GUIDE.md) | 用分级kernel API metadata/statistics/input dumps定位CUDA错误边界，区分graph capture与可回放模块状态。 |
| [debug-distributed-hang](../../library/sglang/.agents/skills/debug-distributed-hang/GUIDE.md) | 按rank日志、watchdog、栈、collective与replay把hang定位到状态分歧，注意同步probe改变时序。 |
| [env-var-conventions](../../library/sglang/.agents/skills/env-var-conventions/GUIDE.md) | 集中EnvField类型化访问与override/alias，明确CLI vs debugenv所有权和迁移默认行为。 |
| [generate-profile](../../library/sglang/.agents/skills/generate-profile/GUIDE.md) | 启动指定服务、按真实工作负载做sanity、采集原生profile并返回文件位置/配置。 |
| [kernel-organization](../../library/sglang/.agents/skills/kernel-organization/GUIDE.md) | 逻辑算子namespace、lazy inventory、runtime状态/插件边界与迁移caller/test/tuning路径全接线。 |
| [kl-consistency-test](../../library/sglang/.agents/skills/kl-consistency-test/GUIDE.md) | 固定token位置比较prefill/decode及warm-prefix状态，正视batch数值与state复用两个条件，探针hitcount有价值。 |
| [large-class-style](../../library/sglang/.agents/skills/large-class-style/GUIDE.md) | 限定三个大类的composition/初始化顺序/窄参数/返回状态与override兼容，便于runtime重构审阅。 |
| [llm-torch-profiler-analysis](../../library/sglang/.agents/skills/llm-torch-profiler-analysis/GUIDE.md) | Existing-trace triage and stage-separated capture, kernel/overlap/fusion candidate reporting across engines. |
| [mechanical-refactor-verify](../../library/sglang/.agents/skills/mechanical-refactor-verify/GUIDE.md) | 用可重放primitive与formatter后byte diff核对纯搬移，分离prepare/move/postpare与人工header审核。 |
| [processor-model-parity](../../library/sglang/.agents/skills/processor-model-parity/GUIDE.md) | 追踪body规范化→renderer→tokenids→OpenAI lowering/流式parser，固定tokenizer revision，明确media/hostfallback与synthetic边界。 |
| [scripted-runtime-notes](../../library/sglang/.agents/skills/scripted-runtime-notes/GUIDE.md) | scripted runtime通过真实scheduler loop/hook驱动状态，避免私有函数在错误phase直接调用。 |
| [sglang-bisect-ci-regression](../../library/sglang/.agents/skills/sglang-bisect-ci-regression/GUIDE.md) | Construct a CI failure timeline and paired SHA experiment with runner identity. |
| [sglang-cherrypick](../../library/sglang/.agents/skills/sglang-cherrypick/GUIDE.md) | Release maintainer preflight, first-parent merge simulation, authorized workflow dispatch and outcome reconciliation. |
| [sglang-multi-graph-check](../../library/sglang/.agents/skills/sglang-multi-graph-check/GUIDE.md) | Configuration-bounded graph lifetime/numerical/coverage investigation, archive inspection and explicit recomputation provenance. |
| [sglang-prod-incident-triage](../../library/sglang/.agents/skills/sglang-prod-incident-triage/GUIDE.md) | Incident baseline collection, preserving replay triggers and routing to crash/hang/profiler tools. |
| [sglang-runtime-context](../../library/sglang/.agents/skills/sglang-runtime-context/GUIDE.md) | Distinguish raw input declarations, resolved bags, per-runner values, runtime flags, resource leases and per-forward state. |
| [speculative-naming](../../library/sglang/.agents/skills/speculative-naming/GUIDE.md) | Consistent internal spec token/block/count naming; distinguish bonus-inclusive acceptance from proposed-draft rates. |
| [write-sglang-test](../../library/sglang/.agents/skills/write-sglang-test/GUIDE.md) | Register CPU/e2e/kernel cases and use defensive server teardown with actual observable assertions. |
| [sglang-diffusion-add-model](../../library/sglang/python/sglang/multimodal_gen/.agents/skills/sglang-diffusion-add-model/GUIDE.md) | Reference-first model port, pipeline-stage selection, conditioning/latent/scheduler contracts, explicit joint modalities and BCG admission. |
| [sglang-diffusion-benchmark-profile](../../library/sglang/python/sglang/multimodal_gen/.agents/skills/sglang-diffusion-benchmark-profile/GUIDE.md) | Native-backend denoise/e2e/profile workflow, same-GPU ABBA applicability and explicit artifacts. |
| [sglang-diffusion-cache-dit](../../library/sglang/python/sglang/multimodal_gen/.agents/skills/sglang-diffusion-cache-dit/GUIDE.md) | Cache-DiT primary/secondary request-over-env injection, mutually exclusive calibrators, matched backend output comparison. |
| [sglang-diffusion-modelopt-quant](../../library/sglang/python/sglang/multimodal_gen/.agents/skills/sglang-diffusion-modelopt-quant/GUIDE.md) | Separate ModelOpt PTQ/export from online quant; checkpoint layout, scale fallback mapping, paired trajectory and performance contracts. |
| [sglang-diffusion-performance](../../library/sglang/python/sglang/multimodal_gen/.agents/skills/sglang-diffusion-performance/GUIDE.md) | Model-specific placement/topology/BCG/cache tuning with exact/eager reference and explicit runtime path evidence. |
