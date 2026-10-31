# KernelAgent

[本地接线和已核对陷阱](../../references/agentic-kernel-workflows.md)。

完整资料包括[README](README.md)、Fuser的子图提取/编排/组合源码、kernel_perf_agent的NCU/roofline/database，以及oink算子与测试。运行完整agent需要原工程的模型provider和目标后端依赖；本包不自动安装或调用它们。

重点追踪 Fuser/pipeline.py → subgraph_extractor.py → orchestrator.py → compose_end_to_end.py → runner.py。组合success在verify=False时仍可能为True；需要verify_passed及对应整图证据。
