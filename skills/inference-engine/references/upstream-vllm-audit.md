# 原有 vLLM 资料逐项复核

2026-10-31 对照官方 vllm-skills 的 6 项、vLLM 主仓库的 9 项。每项的全文、模板和随附脚本都保留；这不是把 15 项压成下面一张表。完整入口见 [catalog](catalog.md)，逐文件摘要见包内来源清单。

| 入口 | 保留的实际能力 | 使用时需要纠正或核对的地方 |
|---|---|---|
| bench-random-synthetic | 随机输入、请求数、并发、结果 JSON | 文中的 10 是示例请求数，不能与 bench-serve 的 1000 混称同一版本默认值；先读当前 CLI。随机请求不能证明质量或真实业务吞吐。 |
| bench-serve | 数据集、QPS/Poisson/gamma 到达、并发上限、逐请求延迟、goodput | 后端和 endpoint 成对指定。带 /v1 的 base URL 与 endpoint 拼接必须按当前客户端验证；保存失败数和实际输出长度。 |
| prefix-cache-bench | 离线重复 prompt 与在线 prefix_repetition 两条路径 | 固定 token 前缀、hash/block 边界、请求顺序与冷/热缓存；两边显式配置 APC。不能仅从重复次数推算 hit rate。 |
| deploy-simple | 环境、启动、ready、API、PID 生命周期 | 原 quickstart.sh 只凭命令存在猜后端，ROCm/CPU 都装同一包；curl -s 会把 HTTP 500 当连通成功，PID 文件也不检查进程身份。该脚本降为参考，实际执行用 [部署步骤](deployment-verified.md)。 |
| deploy-docker | 固定镜像、源码构建、预编译 wheel、ARM64、可选依赖 | 不继承 latest/all-GPU 示例。Python-only wheel 复用前核对基线；C++/CUDA 修改需要相应构建。记录 image digest 与实际加载路径。 |
| deploy-k8s | Deployment/Service、GPU request、共享内存、startup/readiness/liveness | 模板的 CPU/内存/模型是样例。应用前核对 context/namespace、资源名称、端口四处一致、TP 与 GPU requests，复用现有 Secret。Running 仍需真实 API 验证。 |
| check-api-compat | endpoint→protocol→consumer→router，name/type/default/semantics、流与错误 | 规范版本与被测 SDK 分开；vLLM 自身测试不能独立证明兼容。null/omitted、扩展字段、接受后丢弃分别归类。 |
| ci-fails-buildkite | PR/build/job 获取日志、清洗、bisect | .buildkite/scripts 两个脚本与 failures 文档补齐；脚本属于目标 checkout，日志缓存不能代替当前 head 的 CI。 |
| debug-ima | CUDA core dump、cuda-gdb、异步 kernel 定位 | 原入口只是外链。包内已有 [CUDA 排障正文](cuda-core-dump.md)，保留出处与 dump 环境/磁盘要求。 |
| kernel-microbenchmark | CUPTI、graph replay、rotating buffers、多 GPU GEMM+RS | 三个 benchmark 源码保留。cold-L2 是测量选项，不是所有服务的真实缓存状态。多 rank 每样本 MAX 后再取统计；wrapper staging/reset/barrier 计时边界须一致。 |
| model-microbenchmark | bench forward 的 shape 语法、KV jitter、routing、drafter、逐 rank profile | 本次同步最新正文；不含 scheduler、target/rejection sampling 与 CPU step，不能称服务端到端性能。dummy MoE routing 和 graph-vs-eager 语义须单列。 |
| model-recipes | JSON catalog→模型 variant→硬件/strategy→命令与 guide | 保留 read_recipe.sh 和已存 YAML；源 YAML 不等于解析后的可执行命令，generated 不等于 verified。使用目标拓扑核对。 |
| pr-checklist | 核心路径、同步点、实际触发、API/PD 状态流、review threads | 贡献要求属于对应 checkout。行数只是启发，不升级为我们的验收阈值；评论、push 沿用本任务授权。 |
| transformers-upgrade | pin/floor 分开、unvendoring、真实 config/meta-model 双 dump、调用方扫描 | 四个脚本保留。diff_dumps.py 只遍历 before，新增 after 条目可能未显示；打印 error/差异也不保证非零退出。核对两边 repo 集合、重复项及 error，再解释差异。 |
| triton-kernel-writing | specialization/alignment、int64 offset、launch grid、显式中间精度、生成码 | 按当前 Triton 语义判断 barrier 与 alias。覆盖 offset views、mask/大索引、decode/prefill；不要从单一 shape 推广。 |

## 本轮发现的可执行边界

资料中的命令分三类：离线可运行工具、必须在引擎 checkout 中运行的开发脚本、需要模型/GPU/集群的实验。源码和引用齐全不代表后两类已经运行。新知识库查询和打包测试的证据见 [validation](validation.md)。

部署脚本的 HTTP 500 误判、config diff 的新增条目遗漏，以及 KernelAgent 不验证时也返回 success 的行为均有 CPU 复现。对照记录在工作区审计报告中，不用这些结果冒充 GPU 验证。
