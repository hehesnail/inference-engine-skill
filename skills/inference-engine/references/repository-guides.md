# 工程与能力目录

入口按任务选择，不要求一次读取全部来源。18个主仓库包含原调研16个工程及官方vLLM/SGLang源码仓库。745个上游skill入口均有收录或排除记录；302项相关正文及同目录配套资源纳入包。KDA、KernelAgent没有同样形式的根SKILL，另保留主工程资料。

“收录”表示正文与资源可查，不等于每行技术论断已独立复现，更不等于所有外部运行时/GPU实验已执行。具体缺陷、修正与验证见[审计](upstream-vllm-audit.md)、[适配](adaptations.md)、[验证](validation.md)。其余443项入口的元数据筛选记录在 reference-index.json；它们的完整源码仍在服务器研究checkout。

可用 scripts/research_tools.py find 搜索正文入口/主题，并按 --repo 过滤；脚本输出精确revision和实际包内路径。

| 工程 | 收录入口/全部入口 | 具体目录 |
|---|---:|---|
| vllm-project/vllm-skills | 6/6 | [专项列表](repositories/vllm-project--vllm-skills.md) |
| cfregly/gpu-perf-tune | 32/33 | [专项列表](repositories/cfregly--gpu-perf-tune.md) |
| Orchestra-Research/AI-Research-SKILLs | 33/98 | [专项列表](repositories/Orchestra-Research--AI-Research-SKILLs.md) |
| NVlabs/kda | 0/0 | [专项列表](repositories/NVlabs--kda.md) |
| BytedTsinghua-SIA/CUDA-Agent | 1/1 | [专项列表](repositories/BytedTsinghua-SIA--CUDA-Agent.md) |
| NVIDIA/skills | 57/431 | [专项列表](repositories/NVIDIA--skills.md) |
| intel/gpu-ai-skills | 22/23 | [专项列表](repositories/intel--gpu-ai-skills.md) |
| cohere-ai/vllm-skills | 5/5 | [专项列表](repositories/cohere-ai--vllm-skills.md) |
| shen-shanshan/vllm-dev-skills | 16/17 | [专项列表](repositories/shen-shanshan--vllm-dev-skills.md) |
| meta-pytorch/KernelAgent | 0/0 | [专项列表](repositories/meta-pytorch--KernelAgent.md) |
| AMD-AGI/Hyperloom | 17/17 | [专项列表](repositories/AMD-AGI--Hyperloom.md) |
| huggingface/kernels | 5/5 | [专项列表](repositories/huggingface--kernels.md) |
| mit-han-lab/ncu-report-skill | 1/1 | [专项列表](repositories/mit-han-lab--ncu-report-skill.md) |
| BBuf/AI-Infra-Auto-Driven-SKILLS | 11/12 | [专项列表](repositories/BBuf--AI-Infra-Auto-Driven-SKILLS.md) |
| alibaba/atrex-kernel-agent | 19/19 | [专项列表](repositories/alibaba--atrex-kernel-agent.md) |
| NVIDIA/TensorRT-LLM | 35/35 | [专项列表](repositories/NVIDIA--TensorRT-LLM.md) |
| vllm-project/vllm | 9/9 | [专项列表](repositories/vllm-project--vllm.md) |
| sgl-project/sglang | 33/33 | [专项列表](repositories/sgl-project--sglang.md) |

子模块：KDA/KernelWiki、KDA/ncu-report；Atrex/KernelWiki、Atrex/GPU编程教材已按gitlink拉取。KDA两套资料装入完整包；Atrex依赖从其服务器checkout使用。其他内核工具链参考gitlink按目标开发任务加载，不当作已构建依赖。
