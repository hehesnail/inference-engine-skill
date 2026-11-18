# Profiler 与显存容量

## 分析已有 trace

已有日志/trace 是 CPU 工作，按工作区配置选择解析环境，无需安装推理框架或启动 GPU。先识别格式、采集时间、模型/版本、rank、device、stream、prefill/decode/speculative 阶段、图模式和请求形状。显式选择本次输入，避免把共享目录里的历史 trace 混入。

检查 trace 是否真的有 GPU events、完整目标 forward 和可用时间范围。没有事件不能推断为零耗时；缺失 stream、CUPTI 丢记录和 frontend-only trace 都会误导分析。

解释时区分三种量：各 kernel duration 之和、GPU busy interval 的并集、请求墙钟时间。多 stream 重叠会使第一项大于后两项；耗时占比不是可消除的延迟比例。

结合 shape、CPU 调用关联、模型层和真实 dispatch 映射 kernel；不能仅凭名字断言调用路径。看生产者/消费者依赖、通信尾部及 rank skew，再提出 overlap/fusion 候选。无法映射的项保留为未知。

通用解析器覆盖不到自定义模型标记时，可补小型适配器，用 `record_function`/NVTX 区间及 runtime correlation 关联 GPU 事件。说明嵌套区域采用包含子区域的总量还是最深区域归属，保留未匹配项与覆盖率；不要将同一 kernel 在父子区域重复相加。记录采样窗口及其请求阶段，不把短窗口当整轮统计。跨 rank 汇总需校准时间范围，不能直接拼接不同设备的本地时间戳或耗时占比。

graph-off/eager trace 可以帮助定位调用点，graph-on 且充分预热的 trace 用于观察正式执行；只有模型、shape、阶段和 kernel 特征匹配时才转移归因，不把 eager 的计时转移给 graph。CPU `cudaGraphLaunch` 时长不等于 GPU 图执行时长。

## 采集新 trace

先用现有证据判断需要补采什么。确需采集时，按工作区已有授权复现慢场景的输入/输出长度、KV 长度、batch、并行与后端。长上下文 decode 的复现须保留相应的 KV 长度。

核对目标版本的 profiler 接口：vLLM 的 server profiler 配置、worker/frontend 区别及 start/stop；SGLang 的原生采集、分阶段参数、profiler 模式和 frontend 支持。客户端的 `num_steps` 未必等于服务端实际采集窗口。源文件、帮助和实际生成的 trace 共同确认。

为采集结果保留可区分的 run 标识、相关 rank 和原始 trace，避免覆盖或混入旧文件。确认文件属于本次且含预期阶段；文件存在、HTTP 控制返回成功都不是采集有效的充分条件。记录本次临时启动的服务/容器；任务结束后关闭这些临时资源并确认释放，已有常驻服务按本任务安排保留。

报告按问题给出热点与源码归因、暴露的等待/通信尾部、可验证的优化假设；不强制凑齐固定三张表。优化收益须回到无 profiler 的匹配对照测量；NCU replay 可能改变 stream 重叠，不能直接当真实在线时间。

## 显存与 KV 容量

按 rank 和同一时间点对齐权重、KV/状态池、CUDA Graph、runtime/workspace 与其他占用。分别标明日志实测、理论估算和未知项；不同阶段的 `available memory` 之差不一定只对应一个组件。

从最终有效配置、分配日志和实际 GPU 总量推导，区分 GB/GiB、厂商标称容量和可用容量。SGLang 的 `mem_fraction_static` 与 vLLM 的 `gpu_memory_utilization` 不能仅按同一百分比换算；核对各自预算算法及启动后的二次调整。

普通、等 K/V head dimension 的 MHA/GQA 可用每 rank 每 token 近似：`2 × 本 rank 层数 × 本 rank KV heads × head_dim × element_bytes`。先确认 TP 下的 KV head 复制与 PP 层归属。MLA、SWA、压缩 attention、Mamba/混合状态、量化 scale 与 page padding 需读实际存储布局，不能套这个公式。

token 容量除以每请求驻留 token 数只是理想化存储上界；还受 scheduler 请求上限、输出增长、碎片、prefix 共享、临时 workspace 和各 rank 最紧约束影响。它不是满足延迟 SLO 的服务并发。

长文 OOM 先区分配置／容量校验拒绝、权重加载、KV 预留、profile、prefill、decode 和重复运行阶段。历史其他版本的 OOM 不自动归属于当前实现。chunked prefill 主要约束每步处理量，不能保证完整 KV、循环／辅助状态池、全长临时张量或密集 mask 的容量足够；根据失败分配的来源处理，不只缩小 chunk。

设备已用显存、allocator allocated/reserved、预分配 KV／状态池与请求实际活跃状态是不同量。固定比例预留的池会使短请求也显示很高占用，不等于该请求实际需要全部显存。稳定性在同一引擎、请求结束且无在途工作的可比点观察，区分初次编译／缓存扩容后平台与持续增长；短时平台只证明已测范围，没有证据时不宣称长期无泄漏。

## 可复用上游解析器

BBuf 的 trace、容量、逐层分析和图表辅助脚本已连同 import 依赖、参考表和配置一起装入本包。通过[稳定工具入口](tools.md)在选定环境运行；[完整 profiler 指南](../library/bbuf/skills/llm-torch-profiler-analysis/GUIDE.md)与[容量指南](../library/bbuf/skills/llm-serving-capacity-planner/GUIDE.md)提供参数、示例及边界条件。

`offline_tools.py` 只暴露离线模式，拒绝 URL 采集参数并关闭子进程的 GPU 可见性。原生在线采集脚本仍作为完整参考保留，实际调用时沿用工作区已获准的服务、设备与本次资源限制。旧日志匹配不到的字段保留未知，不填入历史默认值。

## 深入具体瓶颈

CPU launch、调度空洞、decode 步预算和 rank skew 读 [Host/GPU 时间线](host-device-analysis.md)；具体 kernel 的 counters、replay、PC stall 和 roofline 读 [NCU](nsight-compute.md)；graph coverage、compile/fusion 与异步消费读 [Graph/同步](cuda-graphs-and-sync.md)。跨引擎、ROCm/XPU 或 PD 链路问题读 [平台与传输](platforms-and-transport.md)。这些是针对当前问题的分支，不要求一起读取。

## 自动 pattern 匹配的证据边界

本包将 fusion 建议的 Name match 与 Candidate 分开：前者是 trace 名称/位置命中，后者只是关键词组同时出现，都不直接证明模型、依赖或 shape 适合该优化。相关 GPU 时间占比大不提升为“已确认”。例如 Gated Delta Net 的 gated 不能只因含 gate 就归到 DeepSeek router GEMM。

混合阶段表的 unclassified 是未能自动归入阶段的那一桶，不是整份 trace 的合计；每桶 share 使用自己的分母。自定义模型可能缺少标准 stage 标记或 Python stack，unresolved 保持未知。先结合本次自定义标记、runtime correlation 和被测源码补归因，不能据 catalog 候选直接修改模型或宣称可消除的延迟比例。
