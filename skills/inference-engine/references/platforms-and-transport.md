# 硬件后端、互连与 PD 接线

用于切换 CUDA/ROCm/XPU、跨设备性能解释、TP/EP 通信和 prefill/decode 解耦排障。专业知识按平台保留；服务器地址、账号、路径和资源授权由当前工作区提供。

## 从真实运行栈选择工具

| 后端 | 首先辨认的对象 | 进一步定位 |
|---|---|---|
| NVIDIA CUDA | GPU SKU、SM/compute capability、driver/runtime、PyTorch/Triton/FlashInfer、加载的 cubin/PTX target | Torch/Nsys 看端到端依赖；NCU 看具体 kernel；sanitizer 看对应 CUDA 正确性问题 |
| AMD ROCm | gfx target、实际 CU/partition、ROCm/HIP、PyTorch/Triton、AITER/CK、RCCL 与真实 dispatch | 使用对应 ROCm 工具查看 kernel、wave/LDS/MFMA、访存和通信；不能套 CUDA metric 名 |
| Intel XPU | PCI BDF/render node、驱动绑定、Level Zero、PyTorch XPU、oneAPI/oneCCL、实际 backend | Torch profiler 的实际 XPU 支持；unitrace 查看 Level Zero/SYCL、queue 与 copy/collective |

容器环境检查要在目标容器/解释器内完成。host 可见设备、`torch.*.is_available()`、节点健康查询、模型能启动和真实推理正确，是不同级别的证据。检查工具缺失为 skipped/unknown，不能变成通过；复现问题时不因某个 skill 的默认命令改用 latest 镜像。

移植已有 CUDA 路径到 Intel GPU 时，从[CUDA→XPU 迁移](../library/intel/plugins/intel-gpu-ai-skills/skills/cuda-to-xpu-migration/GUIDE.md)识别要替换的依赖和调用，在 XPU 环境用 [unitrace](../library/intel/plugins/intel-gpu-ai-skills/skills/xpu-profile-unitrace/GUIDE.md)查 SYCL/Level Zero 的提交和拷贝。已有 NVSHMEM 多进程通信错误时，用 [NVSHMEM 排障](../library/nvidia-skills/skills/nvshmem-troubleshoot-and-report-bugs/GUIDE.md)对齐各 PE 的首个错误和最小复现。

## 保留硬件细节，纠正硬件常量

- H100/H200 的 Hopper 路径与 Blackwell 的 SM100/103 专用路径要分开；TMA/WGMMA 和 TMEM/`tcgen05` 的编译与 layout 约束并不相同。用目标后端与内核支持矩阵确认。
- 同一架构的卡也可能有不同 SM 数、HBM、L2、功耗与频率约束。实际 launch、tile/stage、split 数量应结合当前设备，不能把一台 H200 的测量固化成所有 Hopper 的规则。
- MI355X 是 CDNA4/gfx950，8 个 XCD、每个 32 个 active CU；全卡 L2 与 256 MB Infinity Cache 是不同层。某份上游 skill 写成 CDNA3+ / 32 XCD，不能据此硬编码 swizzle。以 [AMD 官方微架构资料](https://rocm.docs.amd.com/en/docs-10.0.0/reference/gpu-arch/mi350.html)和当前 partition 查询为准。
- ROCm 上核对 FP8 格式（例如 fn / fnuz）、scale 布局、wave size、LDS 和分片；不要只把 CUDA device 字符串替换成 HIP 就宣布迁移完成。
- XPU 的设备选择变量、trace schema 和 backend fallbacks 以版本为准，不能假定 CUDA_VISIBLE_DEVICES 或 CUPTI 配置适用。unitrace 示例是 `unitrace --chrome-call-logging --chrome-kernel-logging python3 workload.py`；先查看该安装的 help 和 device/metric list。采集沿用工作区已有的设备授权，并遵守当前任务指定的设备和时段。
- 自动工具的“Memory stall”分类还需结合流量、吞吐和依赖，不能把所有 memory stall 都称为带宽瓶颈。

## TP / EP 慢与 hang 的分解

把 collective 的参与 rank、顺序、消息大小、dtype、communicator 和 stream 对齐。先分清某 rank 晚到、不同 rank 进入不同分支、通信本身慢、以及不能与计算重叠的尾部。

节点拓扑、device↔NUMA↔NIC 绑定、容器设备挂载、实际选中的 HCA/transport 与启动日志共同解释路径。NCCL/RCCL kernel 的持续时间可能包含等待其他 rank；不能把它全部算成网络传输时间。按已有 [分布式 hang 指南](../library/sglang/.agents/skills/debug-distributed-hang/GUIDE.md)定位最后一致的执行点，再决定需要的诊断。

## PD：分开检查控制、数据和语义

| 层次 | 证据 | 不能据此推出什么 |
|---|---|---|
| 配置/能力 | 实际进程 env、插件/backend、HCA/IB/NVLink、容器可见性 | 环境变量存在不证明走了该 transport |
| 点对点数据链路 | 实际 P/D 两端、size/dtype/设备/方向匹配的 transfer 校验与计时 | 找到 nixlbench 可执行文件不等于完成传输 |
| 请求接线 | P 导出→router 元数据→D 定位/导入→cache ready→计算消费 | endpoint smoke 或合成 HTTP 不证明真实 KV 接线 |
| 运行语义 | cache layout、layer/TP 分片、quant scale、token/page 映射、prefix 命中与增量数据 | 字节拷贝成功不证明解码读的是正确状态 |
| 生命周期 | 完成通知、超时/失败、取消、重试、buffer/page 释放和复用 | 单个请求成功不证明连续请求状态安全 |

NVIDIA Dynamo 的 interconnect check 实现主要探测配置、节点能力和 NIXL 工具存在；它没有自动完成真实 P↔D 传输。缺 `nvidia_peermem` 是线索，GPUDirect 也可能有版本/驱动支持的其他路径；以选中的后端和实测为准，不硬判所有环境都必须有同一模块。

沿一个请求记录生产/转发/消费的字段和值；目标是定位断点。仅有 router 二进制时可用可区分的合成后端值验证 JSON/SSE 字段选择，再在授权内用真实链路确认接线。合成与 GPU 证据分别报告；修正 API usage 不能破坏 scheduler 依赖的内部计数。

## 跨引擎 trace 对照

匹配 model/tokenizer、输入 token、KV 长度、dtype/量化、TP/EP、缓存与 graph 模式。不能按 engine 名强行选择一个赢家；条件不匹配时仍可解释实现差异，限制性能结论。

按层类型和真实模型结构选择代表性层，再用 correlation/边界标记与源码确认算子归组。单靠第 N 次出现或 kernel 名可能把 MoE、混合 attention、fused path 和不同 stream 错配。列出融合边界、转换/拷贝、通信和未知项，并用相同时间口径比较。

没有 GPU step 标记时，CPU 标记平移得到的窗口只能作近似；不要把固定 20 ms 搜索窗或 5% 差异阈值当成所有引擎的规则。针对具体路径提出补接线、启用已有 backend 或新 kernel 的候选，再用匹配实测决定收益。

CUDA/CuTe/cuTile、CPU/ROCm/XPU以及NVSHMEM/DOCA/Dynamo的完整专项见[后端接线](backend-kernel-development.md)。
