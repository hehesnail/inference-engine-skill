---
name: inference-engine
description: Develop, deploy, and optimize LLM inference engines and GPU kernels. Diagnose latency, throughput, memory, or output regressions in kernels, serving, and offline evaluation, especially in vLLM and SGLang.
---

# 推理引擎工作台

把推理引擎问题落实到实际版本、调用路径和可复现结果。LMDeploy 专属接口优先使用工作区相应 skill；共享的模型、状态、kernel 与测量知识可在这里查。纯文案、连接或传输任务直接按工作区处理。

## 按问题取用

从用户要解释或改善的现象选入口。现有证据足以回答时直接使用。若某一步只定位了热点，继续沿对应专题追到能区分原因的对照。

| 当前问题 | 入口 |
|---|---|
| vLLM 源码、模型或服务行为 | [vLLM](references/vllm.md) |
| SGLang 源码、模型或服务行为 | [SGLang](references/sglang.md) |
| 比较部署配置、并发、缓存命中的效果或整套评测耗时 | 从[性能对照](references/benchmarking.md)选基线。离线评测查[生成与评分口径](references/offline-evaluation.md)，多轮调优查[候选记录](references/optimization-workflow.md) |
| 接入模型、改 KV/循环状态或在线更新权重，或定位只在分块、换请求、命中前缀后出现的错误 | [模型接入](references/model-integration.md)。涉及复用或恢复的状态，查[状态生命周期](references/stateful-inference.md) |
| 换精度、量化或后端后分数变了，或要决定是否启用优化 | [数值验收](references/numerical-validation.md)。同一输入的首次差异查[数值漂移](references/numerical-drift-debugging.md)，权重格式与实际算子查[量化接线](references/quantization-engine-contracts.md) |
| 首 token 变慢，kernel 之间有空档，或要用 Torch/Nsys 找请求中的等待 | [Host/GPU 时间线](references/host-device-analysis.md)定位请求关键路径。[Profiler](references/profiling-memory.md)覆盖采集与已有 trace 的解读 |
| 长请求 OOM，或要估算可用显存、KV 容量 | [显存与 KV 容量](references/profiling-memory.md#显存与-kv-容量) |
| 解读单次 launch 的寄存器、访存或 stall 指标 | [NCU 报告](references/nsight-compute.md)先核对计数器能区分哪些机制 |
| 新增或改写 GPU 算子有延迟、吞吐目标，或要解释相对基线的回退 | 用[候选与对照](references/optimization-workflow.md)固定基线、正确性参考与计时边界。遇到 spill、流水线等待或访存异常，按硬件和症状查 [KDA / KernelWiki](references/kernel-knowledge.md)，选能区分源码改动的对照 |
| 新算子、fusion 已写好但模型没变快，或只有图重放出错 | 前者沿[实际引擎接线](references/kernel-engine-integration.md)确认 dispatch、fallback 和整层开销。后者查[图捕获与同步](references/cuda-graphs-and-sync.md) |
| 加草稿模型后接受率提高却更慢，或部分拒绝后输出错误 | [推测解码的成本、已提交 token 和状态回滚](references/speculative-decoding-contracts.md) |
| 相同文本进引擎后 token 不同、约束输出不合法，或流式与最终文本不一致 | [输入输出契约](references/tokenizer-and-structured-output.md) |
| 接入多模态、音频或 diffusion，或排查图文错位或两端 processor 输出不同 | [多模态接线](references/multimodal-inference.md)，[SGLang 专项](references/sglang-specialized-paths.md) |
| 换硬件后走了不同后端，单卡正常但多卡变慢、卡住，或 PD 传输失败 | [平台与通信](references/platforms-and-transport.md) |
| 错误输出、崩溃、hang，或要 review 一段改动 | [排障与 review](references/debugging-review.md)。错误为 CUDA 非法访存时查[崩溃现场与 core dump](references/cuda-core-dump.md) |
| fork 升级、rebase、版本回归 | [版本迁移](references/fork-maintenance.md) |
| 找历史 kernel 实现、PR、硬件技巧，或查某个优化 agent 怎么组织候选和验证 | [KernelWiki 源码与知识检索](references/kernel-knowledge.md)，[五个 agent 工程的工作流](references/agentic-kernel-workflows.md) |
| 查专项、示例或运行离线工具 | [专题目录](references/catalog.md)；[工具](references/tools.md) |

详细指南、模板和脚本保存在 `library/`，通过专题入口或关键词检索按需取用。源工程的固定预算、步骤和示例配置只适用于其场景；技术约束以目标实现及指南中的适用条件为准。按项目查[工程目录](references/repository-guides.md)。固定快照中的路径在本包找不到时，查[上游源码和生成文档的路径对照](references/source-paths.md)。来源与维护见[来源说明](references/sources.md)。

## 实现与证据

- 版本敏感的结论对应实际 checkout、安装包或镜像；核对真正加载的模块、参数与 dispatch，不能用上游示例代替被测代码。
- 状态、布局、精度和计时边界按相关专题及真实调用方核对。区分源码推断、CPU 复现、真实 GPU 推理与性能测量；对照保持工作负载与初态可比。
- 资源使用沿用工作区已有授权及本次明确限制。已有 trace 足以回答时做离线分析。结论依赖真实执行时，在可用且已获准的资源上完成对应复现或测量。

分析任务交代哪一层的输入、状态或输出解释了现象。修复任务沿用调用方依赖的接口和状态约定，从受影响的实际入口验证。优先复用现有测试。用户或仓库要求新增用例，或现有测试捕获不到本次具体回归时，再补相应测试。收尾时给出结论、对应的版本与证据，以及仍影响判断的限制。
