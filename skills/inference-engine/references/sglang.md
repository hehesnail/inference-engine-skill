# SGLang：运行时、kernel 与测试

先确认所用分支或安装镜像。SGLang 的配置、kernel 和测试目录会演进，下面只用于定位；不能因为当前 main 有某接口就假定旧镜像也有。

## 按修改对象找官方 skill

下面的 27 个根目录官方 skills 已保存到[本地专项目录](catalog.md#sglang-专项开发与排障)，完整正文与关联参考都可直接读。版本差异再与目标 checkout 的 `.agents/skills/` 或旧 `.claude/skills/` 核对。

| 修改或问题 | 官方 skill |
|---|---|
| runtime config、ServerArgs、进程全局状态、per-forward state | `sglang-runtime-context`；涉及环境变量再读 `env-var-conventions` |
| speculative decoding 接口与命名 | `speculative-naming` |
| kernel 公共命名空间、注册和文件归属 | `kernel-organization` |
| 轻量 JIT kernel / 大型依赖的 AOT kernel | `add-jit-kernel` / `add-sgl-kernel` |
| 新增或修订测试 | `write-sglang-test` 与目标版本的 `test/README.md` |
| prefill/decode、缓存恢复的 logprob 一致性 | `kl-consistency-test` |
| 分布式卡死 / CUDA crash | `debug-distributed-hang` / `debug-cuda-crash` |
| CI 基础设施、阶段或注册问题 | `ci-workflow-guide` |

`generate-profile` 可用来了解原生采集入口，但其中的固定模型、设备、准确率门槛与清理命令不适合作为个人环境默认值。具体采集按 [profiling-memory.md](profiling-memory.md)。

## 沿实际请求定位

从当前 HTTP/原生入口追到 tokenizer/processor、manager、scheduler、model runner，再到采样、detokenizer 和响应；搜索真实调用边，不用目录顺序代替执行顺序。

主要检索范围为 `python/sglang/srt/` 下的 `entrypoints/`、`managers/`、`model_executor/`、`models/`、`layers/`、`mem_cache/`、`speculative/`、`disaggregation/`。Rust frontend/router 走它自己的入口和协议，不能假定与 Python 服务有相同的 profiler 或管理路由。

配置问题先区分用户原始输入、解析后的有效值、进程/runner 局部状态及一次 forward 的状态。调研时主线已采用 RuntimeContext 与 namespace config bags；旧分支可能不同。先核对发布、读取和 override 的实际实现，避免修改了原始 ServerArgs 却没改到消费方，或把父进程的变化误当作已同步到子进程。

## 模型与 kernel 改动

新模型先列与最接近实现的差异：权重映射、attention/recurrent state、KV 布局、量化、并行分片和协议输出。多模块支持才需要分阶段计划；一个局部修复不必建立完整 Day-0 发布流程。

选择 JIT/AOT 时按当前 kernel 组织约定及依赖决定。轻量路径可参考 JIT，CUTLASS 等大型依赖不机械套同一方案。保留公共包装器、注册、dispatch、fallback 和测试接线。

验证向量化 kernel 时关注真实 stride 与地址对齐，而不只看行长度；量化路径核对舍入、clamp、scale 与存储布局。CUDA Graph 相关改动同时检查 padding、buffer 生命周期和复用。单 op 测试后，还需在相关实际模型路径确认接线。

## 测试与服务启动

优先沿用该版本的 fixture 和注册接口。调研时官方测试指南使用 `CustomTestCase`、可复用 server fixtures，以及 `terminate_and_kill_process_tree` 的优雅清理；它也指出旧测试可能仍保留已淘汰写法。因此，发现邻近样例与当前指南冲突时核对 helper 实现，不能以出现次数最多为准。

测试需断言实际结果、状态转换、错误或协议输出。需要隔离依赖时 mock 边界，避免全局污染；不要把推理行为 mock 掉再宣称验证了它。新测试是否会被 CI 发现，要核对当前注册函数和 suite runner，不能照抄历史 `suite=`、stage 名称或硬件型号。

logprob/KL 问题区分 shape 引起的数值差异与状态/缓存恢复错误。无缓存对照不能证明 prefix cache 正确；常规准确率通过也不能替代缓存状态一致性验证。使用当前测试定义的 workload 和误差准则，不通过放宽阈值掩盖回归。

部署时查实际环境支持 `sglang serve`、`python -m sglang.launch_server` 中哪个入口；benchmark client 也查当前模块的帮助。旧的图模式、DP、benchmark 参数可能已迁移。以实际有效配置和 dispatch 日志记录结果。

来源版本与链接见 [sources.md](sources.md)。

新 kernel、fusion 或 dispatcher 改动按需补读 [kernel 接线](kernel-engine-integration.md)；CUDA Graph 与异步状态问题读 [Graph/同步](cuda-graphs-and-sync.md)。

新增multi-graph、五个diffusion专项和processor parity更新见[专项接线](sglang-specialized-paths.md)，完整33项见[工程目录](repository-guides.md)。

涉及JSON/grammar、speculative与streaming状态时，从 `python/sglang/srt/constrained/`、`speculative/spec_utils.py` 和 `managers/scheduler_components/batch_result_processor.py` 追踪，具体调用路线见 [结构输出契约](tokenizer-and-structured-output.md)。
