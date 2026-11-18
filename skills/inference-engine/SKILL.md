---
name: inference-engine
description: Develop, debug, deploy, profile, and optimize LLM inference runtimes and their kernels, especially vLLM and SGLang.
---

# 推理引擎工作台

把推理引擎问题落实到实际版本、调用路径和可复现结果。LMDeploy 专属接口优先使用工作区相应 skill；共享的模型、状态、kernel 与测量知识可在这里查。纯文案、连接或传输任务直接按工作区处理。

## 按问题取用

已有源码与证据足够时直接推进；需要补充机制、工具或操作细节时，选下面相关入口。各行是独立分支，按问题继续深入。

| 当前问题 | 入口 |
|---|---|
| vLLM 源码、模型或服务行为 | [vLLM](references/vllm.md) |
| SGLang 源码、模型或服务行为 | [SGLang](references/sglang.md) |
| 吞吐、延迟、prefix cache、配置对照 | [Benchmark](references/benchmarking.md)；[离线生成与评分](references/offline-evaluation.md) |
| 模型接入、KV/循环状态、rollout 权重更新 | [模型与状态](references/model-integration.md) |
| 精度、量化、优化开关与验收 | [数值验证](references/numerical-validation.md)；参考实现、舍入与执行差异查[数值漂移排查](references/numerical-drift-debugging.md) |
| Torch/Nsys/NCU、Host/GPU 归因、显存 | [Profiler 与容量](references/profiling-memory.md) |
| kernel、fusion、dispatch、Graph/同步 | [引擎接线](references/kernel-engine-integration.md)；[Graph](references/cuda-graphs-and-sync.md) |
| 多轮候选与 agent 优化机制 | [候选与对照](references/optimization-workflow.md)；[Agent 工程](references/agentic-kernel-workflows.md) |
| Tokenizer、模板、结构化或流式输出 | [输入输出契约](references/tokenizer-and-structured-output.md) |
| 多模态、音频、diffusion | [多模态接线](references/multimodal-inference.md)；[SGLang 专项](references/sglang-specialized-paths.md) |
| 硬件后端、TP/EP、PD 传输 | [平台与通信](references/platforms-and-transport.md) |
| 错误输出、崩溃、hang 或代码 review | [排障与 review](references/debugging-review.md) |
| fork 升级、rebase、版本回归 | [版本迁移](references/fork-maintenance.md) |
| KDA 技术知识、历史实现与代码证据 | [KernelWiki](references/kernel-knowledge.md) |
| 查专项、示例或运行离线工具 | [专题目录](references/catalog.md)；[工具](references/tools.md) |

详细指南、模板和脚本保存在 `library/`，通过专题入口或关键词检索按需取用。源工程的固定预算、步骤和示例配置只适用于其场景；技术约束以目标实现及指南中的适用条件为准。来源与维护见 [sources](references/sources.md)，按工程检索见 [工程目录](references/repository-guides.md)。

## 实现与证据

- 版本敏感的结论对应实际 checkout、安装包或镜像；核对真正加载的模块、参数与 dispatch，不能用上游示例代替被测代码。
- 状态、布局、精度和计时边界按相关专题及真实调用方核对。区分源码推断、CPU 复现、真实 GPU 推理与性能测量；对照保持工作负载与初态可比。
- 资源使用沿用工作区已有授权及本次明确限制。已有 trace 足以回答时做离线分析。结论依赖真实执行时，在可用且已获准的资源上完成对应复现或测量。

分析任务交代哪一层的输入、状态或输出解释了现象。修复任务沿用调用方依赖的接口和状态约定，从受影响的实际入口验证。优先复用现有测试。用户或仓库要求新增用例，或现有测试捕获不到本次具体回归时，再补相应测试。收尾时给出结论、对应的版本与证据，以及仍影响判断的限制。
