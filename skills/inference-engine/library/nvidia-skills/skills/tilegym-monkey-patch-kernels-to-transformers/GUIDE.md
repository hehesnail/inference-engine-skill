> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[NVIDIA/skills@25340619336e](https://github.com/NVIDIA/skills/blob/25340619336e89bcc40365c28bfe9e790430468e/skills/tilegym-monkey-patch-kernels-to-transformers/SKILL.md)，2026-10-31 核对。

# Integrate and create cuTile kernels into 🤗 Transformers
The main purpose of TileGym project is to provide performant kernels for LLM training and inference. We will integrate proper kernels available in TileGym project to LLM models provided by Hugging Face `transformers` library to validate end-to-end functional correctness and performance improvements. Instead of modifying `transformers` source code, we will take a non-intrusive monkey-patch approach: We will replace certain modules/classes/methods in `transformers` library that implement the Transformer model we would like to integrate, such that at model instantiation, that model's core components will be replaced by TileGym implementations. At runtime the model will actually invoke TileGym kernels under the hood. In addition, we will follow an auto-research-style agent harness loop to create and integrate new cuTile kernels to the target model to improve kernel coverage and end-to-end throughput.

## Instructions
This is for human readers: Simply prompt your favorite AI Agent with skill name and target model ID. E.g.,:
```Claude/CodeX
Hi, please /monkey-patch-kernels-to-transformers Qwen/Qwen3.5-0.8B.
```
The Agent might ask you several questions. Make clarifications and give a go confirmation.

## Workflow
1. Prepare experiment environment. Follow [environment-setup.md](references/environment-setup.md)
2. Integrate existing TileGym kernels to the target model. Follow [kernel-integration.md](references/kernel-integration.md)
3. Autonomously create new cuTile kernels for uncovered PyTorch code. Follow [auto-kernelize.md](references/auto-kernelize.md)
   * Feel free to add new cuTile kernels with constraints in mind
   * Do not stop until meet auto-kernelize loop stop conditions
4. Summarize and report

## Disciplines
This is for AI Agents executing this workflow.

### Kernel inventory
Reusable transformer-local kernels must be represented with FlashInfer-Bench-style Definition and Solution metadata. Follow [kernel-inventory-schema.md](references/kernel-inventory-schema.md) when researching compute requirements, inventorying existing kernels, proposing candidates, or creating new generated kernels.
