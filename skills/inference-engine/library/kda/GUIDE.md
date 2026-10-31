# Kernel Design Agents

先读[本地工作流与接线](../../references/agentic-kernel-workflows.md)和[知识库](../../references/kernel-knowledge.md)。

- [完整流程](docs/agent-flow.md)
- [任务契约模板](prompts/basic-flow.md)：按现有任务信息填写，不强制重复计划审批。
- [HF导出协议](docs/huggingface-export.md)与[导出脚本](scripts/export_huggingface.py)
- [原始CPU测试](tests/test_export_huggingface.py)：包含fake builder与真实builder条件性测试。
- [完整KernelWiki](../kernelwiki/GUIDE.md)；[锁定的NCU](../ncu-kda/GUIDE.md)
- [许可证与第三方来源](THIRD_PARTY_NOTICES.md)

README中的插件安装是原工程的宿主方式；使用本工作台不需要安装Humanize或启动其他agent。
