# 后端、kernel 与通信的具体工作流

按目标后端和交付形式选择实现资料。工程版本及 NVIDIA、Intel、Hugging Face、TensorRT-LLM 的专项目录见[工程目录](repository-guides.md)。

## CUDA / Triton / CuTe / cuTile

先从引擎实际 dispatch 查 wrapper 和 kernel：输入 shape/stride/dtype、原位 alias、输出分配、custom op schema、fake/meta、stream、graph、fallback，之后才选实现语言。

NVIDIA TileGym 的接线是公共 op→backend implementation→包初始化注册→测试/benchmark；仅新增函数不保证 dispatch 使用。cuTile 的 exhaustive_search 将 args_fn、grid_fn、hints_fn 分开，选中 configuration 与 replace_hints 后的 kernel 对象都缓存，再直接 ct.launch。缓存键要包括设备架构、dtype、shape/stride 与影响语义的参数；不能用首次机器的结果覆盖其他设备。

原位算子搜索用 scratch/split buffers，最终运行才写真实输入；搜索前恢复数据，计时前完成 JIT。保留固定配置 fallback，检查 DISABLE_AUTOTUNE 路径。tile、occupancy、num_ctas 是否值得搜索由算子和瓶颈决定。上游“所有 kernel 都应 autotune”“含原配置就保证不退化”“首次测速快就不再验证”“host wrapper 不影响性能”均不作为本工作台规则；启动成本、测量噪声和 wrapper 开销都要算。8/30 个配置只是来源中的预算样例。

CuTe/Blackwell 重点读 TMEM、tcgen05、TMA、mbarrier、warp specialization、2-CTA、CLC 和 WGMMA 迁移；SM90/SM100/SM120 分开判适用性。精确例子与已发现错误见 [KernelWiki](kernel-knowledge.md)。没有这些指令的 GPU 使用对应后端路径，不因本轮机器不同删掉专项知识。

## Hugging Face kernel-builder

将已验证实现交付为可安装的 kernel 时，按源语言和后端选择[五种构建指南](repositories/huggingface--kernels.md)。已有 CuTe-DSL Python/JSON 候选可按 [KDA 本地导出](../library/kda/docs/huggingface-export.md)提供入口、release 配置及独立输出目录，并在目标环境用 `kernels.get_local_kernel` 加载产物、以相同输入核对输出。含原生 CUDA/C++ 的扩展走 [CUDA 构建及 dispatcher 注册](../library/hf-kernels/kernel-builder/skills/cuda-kernels/GUIDE.md)。导出环境需要对应版本的 `kernel-builder`。

五套 CPU/CUDA/ROCm/Triton/XPU guide 连同 references 与项目模板保留。它们解决可分发扩展的项目布局、构建变体、依赖、operator registration、测试和加载，不替代引擎的 custom op/fake/graph 接线。

注意 Stable ABI、torch binding、JIT 与 AOT 是不同边界。先确认目标渠道所用 kernel-builder/kernels 版本、Python/PyTorch/CUDA/ROCm ABI 和架构列表，再选择构建方式；不把其渠道要求推广到全部 vLLM 扩展。包导入成功之后仍需实际运行目标算子与 fallback。MI355X 使用 CDNA4/gfx950；资料中的 CDNA3、32 XCD、把 Infinity Cache 当 L2 等表述已标记纠正，硬件数据以 AMD 架构文档为准。

## Intel XPU

按 xpu-discover→runtime-preflight→model-type/config/fit→container或现有环境→vllm/sglang run→bench/profile 路由。22 项包括 vLLM-Omni、llama.cpp、Torch 单算子、CUDA→XPU 迁移、unitrace 与部署诊断，不只保留 vllm serve 命令。

核对 Level Zero/驱动、可见设备、PyTorch XPU、引擎 wheel、编译工具与容器 device/group 权限；import 成功不证明 kernel backend 可运行。容量估算区分权重、量化 scale、KV/Mamba state、activations/workspace、并发与上下文长度。性能用实际生成长度、TTFT/TPOT/吞吐和同步边界，不把 torch.cuda 工具套到 XPU。unitrace/torch profiler 先确认活动类型和时间单位，异步命令的 wall time 需同步或事件解释。

迁移从 CUDA 专属 API、stream/event、custom op、编译 flag、layout/精度行为逐项映射；保留 CUDA 路径和 unsupported fallback，测试真的进入 XPU 分支。AMD 与 XPU 是条件性能力，不写死某台 NVIDIA 机器。

## NVSHMEM / RDMA / PD

NVSHMEM 9 项保留安装、入门、文档、transport 选择、NIC/PE 映射、TMA、采集、调优和故障处理。把三件事分开：操作是否支持、transport 是否实际选中、目标 workload 是否更快。

- IBGDA 是 GPU 数据路径，仍要确定 host transport；GPUNetIO GDAKI 与 IBGDA 不同时启用。UCX/libfabric 的 provider 控制不能伪装成通用 HCA 精确映射。
- NIC 映射使用目标计算节点的端口状态、IB/RoCE link layer、GPU PCI BDF、PE→GPU 实际绑定和本次 NVSHMEM 版本。不要拿登录节点的 topology、逻辑 GPU 序号或 HCA 数量代替这些证据。
- HCA_LIST 与 HCA_PE_MAPPING 选择机制分开；版本相关 BLOCK/multi-port 行为查随附矩阵，再对目标 release 核实，不把“3.8 or later”当永久承诺。
- NVSHMEM TMA 的作用域是支持的 GPU peer load/store 路径和操作，常见于 NVLink；不是让 IB/RoCE 自动变成 TMA。保留注册、dynamic SMEM sizing、alignment、CTA control flow、completion 和 lifetime；不能仅因为 registration 成功就报告 TMA 已使用。
- DOCA 的 RDMA/verbs/GPUNetIO/perftest 保留作为底层诊断。Dynamo 的 interconnect→recipe→router→troubleshoot 接回服务链；工具存在、注册成功和真正双端传输通过分别记录。

PD 排查沿请求 admission、P cache、transfer registration/export、router metadata、D import/read、生成与释放走一遍。NIC link up、NIXL 可 import、HTTP 200 都不是 KV 正确交接证据。通信微测、kernel overlap 和真实服务吞吐各自报告。

## 边缘与调度平台

Jetson/DGX Spark 的共享内存、CPU/GPU 竞争、功耗/时钟、offload 与 speculative decoding 指南保留；其示例不能套为 H200 参数。Run:ai 的 quota、工作负载与节点健康诊断是对应平台的可选入口。Nemo Megatron 的 16 份性能资料作为训练侧技术参考，采用 overlap、MoE dispatcher、parallelism 机制时重新核对推理数据流，不把训练验证直接算作推理收益。
