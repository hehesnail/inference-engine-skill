# NVIDIA/TensorRT-LLM

固定提交：[f405e10b730ea953961ecf5719242c581527587a](https://github.com/NVIDIA/TensorRT-LLM/tree/f405e10b730ea953961ecf5719242c581527587a)。

先读本地适配：[性能和框架专项](../inference-optimization-cases.md)。

当前采用 35/35 项。采用项正文逐篇语义审阅；相关源码和附属资源的核对范围单列，不能推广成整仓每行已验证。技术修订与限制见 [语义核对](../semantic-audit.md)。

| 专项 | 用途 |
|---|---|
| [exec-env-check](../../library/trtllm-skills/.claude/skills/exec-env-check/GUIDE.md) | 区分本机、Docker、Slurm执行位置，保留硬件与运行前提检查。 |
| [exec-local-compile](../../library/trtllm-skills/.claude/skills/exec-local-compile/GUIDE.md) | 精确源码/架构/容器下的增量编译、链接与import定位。 |
| [exec-local-docker](../../library/trtllm-skills/.claude/skills/exec-local-docker/GUIDE.md) | 容器执行的退出码、日志、构建先决条件与hang监测。 |
| [exec-local-slurm](../../library/trtllm-skills/.claude/skills/exec-local-slurm/GUIDE.md) | Slurm allocation/step生命周期、容器复用和sacct终态记录。 |
| [exec-remote-slurm](../../library/trtllm-skills/.claude/skills/exec-remote-slurm/GUIDE.md) | 保留SSH远端构建与Slurm调度、固定镜像/架构、轻量日志取证方法。 |
| [exec-slurm-compile](../../library/trtllm-skills/.claude/skills/exec-slurm-compile/GUIDE.md) | Slurm CPU构建队列、enroot import、目标wheel与ABI产物验证。 |
| [kernel-cute-writing](../../library/trtllm-skills/.claude/skills/kernel-cute-writing/GUIDE.md) | CuTe布局、TMA/MMA流水线、框架custom op与编译缓存。 |
| [kernel-tileir-optimization](../../library/trtllm-skills/.claude/skills/kernel-tileir-optimization/GUIDE.md) | 现有Triton切换TileIR backend与配置搜索、同输入PTX对照。 |
| [kernel-triton-writing](../../library/trtllm-skills/.claude/skills/kernel-triton-writing/GUIDE.md) | Triton分块/掩码/累加dtype、autotune与自定义op测试。 |
| [perf-analysis](../../library/trtllm-skills/.claude/skills/perf-analysis/GUIDE.md) | 性能指标→瓶颈假设→相关证据→改进候选。 |
| [perf-host-analysis](../../library/trtllm-skills/.claude/skills/perf-host-analysis/GUIDE.md) | nsys host prep暴露/隐藏、请求阶段与多rank归因。 |
| [perf-host-optimization](../../library/trtllm-skills/.claude/skills/perf-host-optimization/GUIDE.md) | 围绕实际PyExecutor热点做CPU行级分析、hoist/cache/减少launch后端开销。 |
| [perf-instability-cookbook](../../library/trtllm-skills/.claude/skills/perf-instability-cookbook/GUIDE.md) | 用运行方差、warmup、JIT/GC/调度与通信案例定位推理时延抖动。 |
| [perf-nsight-compute-analysis](../../library/trtllm-skills/.claude/skills/perf-nsight-compute-analysis/GUIDE.md) | 用NCU逐kernel的计数器、roofline和stall定位算子瓶颈。 |
| [perf-nsight-systems](../../library/trtllm-skills/.claude/skills/perf-nsight-systems/GUIDE.md) | 捕获真实推理稳态时间线、Python launch与GPU/通信依赖。 |
| [perf-optimization-casebook](../../library/trtllm-skills/.claude/skills/perf-optimization-casebook/GUIDE.md) | 用已修性能案例查候选机制和可能受益shape，而非重建问题历史。 |
| [perf-optimization](../../library/trtllm-skills/.claude/skills/perf-optimization/GUIDE.md) | 把profiling、候选改动、kernel与backend验证组织成推理优化闭环。 |
| [perf-regression-cookbook](../../library/trtllm-skills/.claude/skills/perf-regression-cookbook/GUIDE.md) | 用已解决回归的触发条件/定位路径缩小当前推理性能回归范围。 |
| [perf-torch-cuda-graphs](../../library/trtllm-skills/.claude/skills/perf-torch-cuda-graphs/GUIDE.md) | 为推理decode小算子和调度launch减少CPU开销，核对固定buffer、bucket、流、pool和RNG。 |
| [perf-torch-sync-free](../../library/trtllm-skills/.claude/skills/perf-torch-sync-free/GUIDE.md) | 减少推理循环CPU/GPU值回传等待，保留终止条件、输出与索引契约。 |
| [perf-workload-profiling](../../library/trtllm-skills/.claude/skills/perf-workload-profiling/GUIDE.md) | 保留 warmup、预分配、事件记录和独立 profiler 归因；接回推理引擎时明确定时窗口、完成边界、跨流依赖，并恢复 KV/SSM/RNG 等可变状态。 |
| [trtllm-case-executor](../../library/trtllm-skills/.claude/skills/trtllm-case-executor/GUIDE.md) | 保留精确命令、job manifest、资源推导、镜像身份、超时与结果分类；用于 vLLM/SGLang 时接既有 SSH/执行器并保留实际源码和实验输入身份。 |
| [trtllm-code-contribution](../../library/trtllm-skills/.claude/skills/trtllm-code-contribution/GUIDE.md) | 保留官方贡献流程、DCO、邻近风格、调用方/权重布局/并行维度检查、精简测试与可审阅差异；本地修复不自动转为外部发布。 |
| [trtllm-codebase-exploration](../../library/trtllm-skills/.claude/skills/trtllm-codebase-exploration/GUIDE.md) | 保留先沿调用路径、状态、权重布局和 dispatcher 查已有抽象，再决定复用的工程方法；vLLM/SGLang 使用其真实路径。 |
| [trtllm-flashinfer-upgrade](../../library/trtllm-skills/.claude/skills/trtllm-flashinfer-upgrade/GUIDE.md) | 保留依赖 pin/lock/attribution 一致性、目标版本选择、调用点兼容与可追溯更新；接回 vLLM/SGLang 的 FlashInfer 变更时换成各自实际锁定和使用路径。 |
| [trtllm-model-onboard-multimodal](../../library/trtllm-skills/.claude/skills/trtllm-model-onboard-multimodal/GUIDE.md) | 保留 processor→placeholder/position→批量 encoder→按 item 分拆→缓存→chunk 融合→LLM 的数据流，以及权重 mapper、请求清理与独立 HF 比较；这比孤立“能跑图像”更能支撑推理引擎接线。 |
| [trtllm-moe-develop](../../library/trtllm-skills/.claude/skills/trtllm-moe-develop/GUIDE.md) | 保留 ConfigurableMoE、scheduler、strategy、leaf 的所有权边界，导入核 ABI、路由索引、量化状态与迁移参数成组一致；接回引擎时先固定对象/索引空间和 collective 参与者。 |
| [trtllm-serve-config-guide](../../library/trtllm-skills/.claude/skills/trtllm-serve-config-guide/GUIDE.md) | 保留按目标 latency/throughput 与真实配置找起点，再验证运行；接回 vLLM/SGLang 时用实际调度 token budget、rank-local 权重/KV/图捕获预留估算容量，再测 SLO。 |
| [trtllm-test-script-builder](../../library/trtllm-skills/.claude/skills/trtllm-test-script-builder/GUIDE.md) | Generate reviewable Docker/Slurm execution manifests with resolved checkpoint mounts, rank layout, image architecture, MPI variant and separate allocation/runtime identities. |
| [trtllm-test-specialist](../../library/trtllm-skills/.claude/skills/trtllm-test-specialist/GUIDE.md) | Classify module, model, perf-sanity, benchmark and accuracy work; preserve explicit inputs; construct workload-specific commands; track terminal job status and honest test counts. |
| [trtllm-visualgen-bench](../../library/trtllm-skills/.claude/skills/trtllm-visualgen-bench/GUIDE.md) | Reproducible VisualGen HTTP workload snapshots, model/server identity, warmup boundaries, request-level latency breakdown, retained generated media and audio validation. |
| [trtllm-visualgen-lpips-golden-refresh](../../library/trtllm-skills/.claude/skills/trtllm-visualgen-lpips-golden-refresh/GUIDE.md) | Intentional golden refresh with failure triage, family-specific relative zip members, byte checksums, source/package/checkpoint provenance and affected-test revalidation. |
| [visual-gen-component-test](../../library/trtllm-skills/.codex/skills/visual-gen-component-test/GUIDE.md) | L1 operation/L2 component paired experiments with actual shared noise tensors, canonical weights, native intermediate outputs, feature activation evidence and explicit zero-norm handling. |
| [perf-analyze](../../library/trtllm-skills/agent-flow/.claude/skills/perf-analyze/GUIDE.md) | Optional benchmark/projection/nsys/ncu/report campaign with separated operating points, provenance, bounded profiling and restartable recorded stages. |
| [perf-optimize](../../library/trtllm-skills/agent-flow/.claude/skills/perf-optimize/GUIDE.md) | Bounded optimization experiments with baseline/current-best comparisons, measurable acceptance, rollback, runtime identity, independent final verification and evidence-linked remaining work. |
