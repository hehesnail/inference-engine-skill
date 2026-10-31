# vLLM：先定位当前实现

下列路径是检索入口，不是跨版本接口保证。用当前 checkout 的 `rg --files`、符号搜索和调用方确认；旧版未必采用同一目录布局。

## 源码与官方 skills

| 问题 | 首先定位 |
|---|---|
| OpenAI 请求、streaming、usage | `vllm/entrypoints/openai/`、`vllm/entrypoints/serve/engine/`，以及 `tests/entrypoints/` |
| 调度、batch、KV/prefix cache | `vllm/v1/core/`、`vllm/v1/engine/` 与 `tests/v1/` |
| 模型执行、attention、采样、spec decode | `vllm/v1/worker/`、`vllm/v1/attention/`、`vllm/v1/sample/`、`vllm/v1/spec_decode/` |
| 权重、模型适配、量化、MoE | `vllm/model_executor/model_loader/`、`models/`、`layers/` 与当前模型贡献文档 |
| kernel 和编译优化 | 具体 dispatch 到的 kernel、`vllm/compilation/`、`csrc/`、相应 correctness suite 与 `benchmarks/kernels/` |

官方两套资源已完整收录：[开发指南](catalog.md#vllm-开发)与[部署/benchmark 指南](catalog.md#vllm-部署与服务-benchmark)。本地保留各指南的完整正文、示例和配套脚本。目标 checkout 若只有 `.claude/skills/`，先确认它是否为兼容入口。

按问题补读相应指南的相关部分；接口和行为以当前 checkout 为准：

- `check-api-compat`：判断兼容 API 的字段、默认值或流事件行为。
- `debug-ima`：异步 CUDA 报错，需要定位 faulting kernel。
- `kernel-microbenchmark` / `model-microbenchmark`：相应粒度的性能实验。
- `triton-kernel-writing`：确实编写 Triton kernel 时。
- `pr-checklist` / `ci-fails-buildkite`：分别在准备 PR、诊断该 CI 时。

不要让独立部署 skill 的安装步骤覆盖正在复现的镜像；也不要把最新主仓库的运行参数回填到旧 wheel。

## 部署与环境

复用用户指定的环境。记录 image digest 或源码 SHA、包版本和源码加载位置、模型/tokenizer revision、实际启动命令。环境准备遵循目标 checkout 的贡献说明；例如其指定 `uv`/venv 时使用对应环境，不把这个要求扩展到外部日志解析脚本。

新部署从模型所需的最小配置开始。已部署实例先读启动日志与有效配置；一个启动成功的服务只证明加载和基本链路，不证明准确率或容量。

在目标环境查 `vllm serve --help` 和 `vllm bench serve --help`。若帮助被分组，再查该版本提供的完整帮助（例如 `--help=all`）。据此生成命令，不依赖 skill 中的参数表。端口、模型名、网络绑定与设备沿用本任务配置。

## API 与模型修改

接入新模型或自定义状态时先用 [model-integration.md](model-integration.md) 沿完整执行链核对；数值与候选开关的证据口径见 [numerical-validation.md](numerical-validation.md)。只选本次实际影响的路径，不将清单变成全量测试要求。

API 兼容性要同时核对实现与目标协议的官方规范/对应 SDK 版本：字段名、类型、默认值、语义，以及缺字段与显式 null 的区别。声明了字段但消费链路丢弃它，同样可能是错误。引擎扩展字段与协议冲突分别说明。

沿实际请求检查解析、模板/tokenizer、引擎输入、输出处理和 JSON/SSE；不要仅靠 Pydantic 对象构造或 mock 调用次数证明 HTTP 行为正确。流式协议的终止事件按具体 endpoint 核对，不把 Chat Completions 的终止规则套给 Responses。

模型适配先比较 config、权重名称/分片、位置编码、attention/KV 布局和前后处理；复用最接近且契约相同的实现。专用 kernel 的 shape、dtype、量化 scale、padding 与 fallback 都属于契约。确认真实模型 dispatch 到候选路径，再解释测试覆盖和性能。

## 验证与贡献

先读修改路径适用的 AGENTS、邻近测试和测试说明，按行为补最小有效回归；公开入口的行为要有对应入口验证。kernel correctness 与微基准分开保存，微基准改善不能单独证明服务变快。

准备向 vLLM 提交 PR 时，重新读取该 SHA 的贡献政策、PR 模板和 `pr-checklist`。当前调研版本要求检查重复工作、披露 AI 辅助并由提交人理解和审阅；这些要求在实际对外提交阶段落实，不把本地排障提前变成审批流程。

来源版本与链接见 [sources.md](sources.md)。

本地还保存了 [模型 recipes 源配置](../library/vllm/recipes/README.md)；它们带条件与 variant，查找方式和解析注意事项见 [model-recipes 完整指南](../library/vllm/.agents/skills/model-recipes/GUIDE.md)。CUDA IMA 的实际采集与 cuda-gdb 操作见 [本地操作说明](cuda-core-dump.md)。

新 kernel、fusion 或 dispatcher 改动按需补读 [kernel 接线](kernel-engine-integration.md)；长期定制分支升级读 [fork 维护](fork-maintenance.md)。

官方6项部署/压测与9项开发指南的逐项复核和可执行边界见[原vLLM审计](upstream-vllm-audit.md)；部署采用[已核对步骤](deployment-verified.md)。
