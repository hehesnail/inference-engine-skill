# 当前吸收范围与证据

此前 16 仓库的研究只抽取了部分方法，不能称作完成了工程吸收；尤其没有把 KDA 的知识依赖及多数完整技能资源接进当前工作台。本次保留这一差异，不沿用旧报告中“仅作主题线索”的结论。

当前覆盖 18 个主工程、745 个标准技能条目，逐项登记采用或排除理由；302 项推理工程相关技能正文已完整读取并语义核对，KDA 48 页技术正文另有记录；具体问题、保留能力与适用条件见[本轮核对](semantic-audit.md)。逐仓数量、正文和未采用清单见 [工程目录](repository-guides.md)。KDA/KernelAgent 等非标准 skill 工程另有专门入口。

吸收分成三个可检查层次：

1. 资源完整：GUIDE 正文、相关 references、脚本、模板、配置与许可证有固定来源；缺失的上游路径明确标注。全部条目有清单，不等于逐行验证所有工程代码。
2. 工程接线：把任务契约、知识检索、基线、候选、真实验证、引擎 dispatch、组合验证、恢复与交付落实到 [agent 工作流](agentic-kernel-workflows.md)、[后端开发](backend-kernel-development.md)、[优化案例](inference-optimization-cases.md) 和 [SGLang 专项](sglang-specialized-paths.md)。
3. 实际验证：只把真正运行的脚本和用例列为通过；CPU 工具及部分 H200 路径已经实跑，SGLang、其他硬件后端、集群和完整外部 agent 尚未实跑，见 [验证记录](validation.md)。上游的 verified、success、report accepted 和配置预期收益不能替代当前测量。

原有官方 vLLM 15 项已全文复核，并对关键 helper 的结果语义做 CPU 复现。新工程侧检查了 KDA 导出与知识查询、KernelAgent 组合、Hyperloom 配置/候选/恢复/质量分支、Atrex light/legacy 的记录与接受边界、cfregly 指标/选择/证据模块，以及硬件与多后端专项的关键契约。原始阅读日志只说明读取过的范围，不把因输出截断而未见到的内容算作已精读。

443 个未采用入口只按名称与描述做范围筛选，记录为本轮未纳入，不能断言其正文无用；训练机制资料中确实能帮助分析通信、MoE、内存的部分保留并标注适用条件。没有以“通用”为由排除 Blackwell、ROCm、XPU、diffusion、NVSHMEM 或具体算子知识。
