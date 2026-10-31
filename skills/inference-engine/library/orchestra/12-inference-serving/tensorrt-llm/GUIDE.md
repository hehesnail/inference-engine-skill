# TensorRT-LLM 推理与运行时接线

明确当前使用 PyTorch backend、TensorRT engine 还是兼容服务接口；它们的构建、配置和输出契约不同。目标 wheel/容器、GPU 架构和模型支持矩阵决定路径，不把历史 release candidate 的示例当成 latest。

## 加载、输出与量化
LLM API 的 generate 返回 RequestOutput 或其列表；文本在候选输出中，例如检查 `request_output.outputs` 后取相应候选的 `text`，不是假定顶层 output.text。流式、beam 和多请求时逐层保留请求身份及完成状态。

FP8/INT4 等量化不仅是给模型 dtype 写个字符串；还涉及 checkpoint、scales、quant recipe、硬件指令和实际 kernel 支持。A100 与 Hopper/Blackwell 的低精度能力不同，不能承诺 A100 有 Hopper FP8 Tensor Core 加速。按 [量化契约](../../../../references/quantization-engine-contracts.md) 核对。

## 性能与部署
host launch、同步、graph、调度、KV/prefix、GEMM/MoE、通信与输出处理分阶段定位，保留相同实际 workload 和错误数。build-time capacity 不等于 runtime 当前 batch。首轮加载/编译/图捕获与稳态计时分开；历史 tok/s 表不作当前预期。

TP/PP/EP/DP 的 world/rank 与进程启动器一致，检查多节点 rendezvous 与网络，不凭环境变量名套另一引擎。具体专题优先使用本包已逐项审阅的 [TRT-LLM 工程指南](../../../../references/repository-guides.md) 中35个 upstream 专项；原 Orchestra refs 的历史配置须对照所用版本，不拼接不同 backend 的字段。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `12-inference-serving/tensorrt-llm/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [multi-gpu](references/multi-gpu.md)
- [optimization](references/optimization.md)
- [serving](references/serving.md)

完整历史正文及本次具体勘误保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于追溯；运行规范使用本页与专题。
