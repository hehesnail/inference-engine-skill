> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[shen-shanshan/vllm-dev-skills@1ccafeebf71b](https://github.com/shen-shanshan/vllm-dev-skills/blob/1ccafeebf71b5d922f351a0cd009a258b7474fac/skills/vllm-model-tutorial/SKILL.md)，2026-10-31 核对。

# vLLM Model Tutorial Generator

Generate a comprehensive model technical tutorial document for a given model supported by vLLM.

## Workflow

### Step 1: Identify the Model

Extract the model name from the user's request. Normalize common aliases:

- "Qwen3-VL" / "Qwen3VL" → Qwen3-VL
- "Qwen2.5-VL" → Qwen2.5-VL
- "DeepSeek-V3" / "DSv3" → DeepSeek-V3
- "InternVL3" / "InternVL 3" → InternVL3
- "Llama 4" → Llama 4
- "GPT-OSS" → GPT-OSS

If the model name is ambiguous, ask the user to clarify.

### Step 2: Research the Model

Gather information from multiple sources. This is the most critical step — thorough research determines document quality.

**2a. Find Technical Reports and Papers**

Search for the model's official technical report, paper, or blog post:
- Use WebSearch: `"{model_name} technical report arxiv"` or `"{model_name} paper"`
- Use WebFetch to read the paper/report and extract architecture details, innovations, benchmarks
- For model series: also find reports for predecessor models to build the evolution timeline

**2b. Gather Model Family Information**

Build the model family comparison context:
- Search for the full model series evolution (e.g., Qwen-VL → Qwen2-VL → Qwen2.5-VL → Qwen3-VL)
- For each variant: collect parameter counts, release dates, key innovations, performance benchmarks
- Find HuggingFace and ModelScope links for each variant (search `huggingface.co/{model_id}`)
- Collect technical report / paper links for each variant

**2c. Analyze Model Architecture**

Extract detailed architecture information from papers, docs, and the LLM Architecture Gallery:
- Overall architecture design (encoder-decoder, decoder-only, cross-attention)
- Key components: attention mechanism (MHA/GQA/MQA/MLA), FFN type (dense/MoE), normalization, activation
- For VLMs: ViT architecture, visual token projection, multimodal fusion strategy
- Context length, vocabulary size, hidden dimensions, layer counts
- Special tokens, chat template, generation config
- Reference: `https://sebastianraschka.com/llm-architecture-gallery/` for comparative context

**2d. Explore vLLM Source Code**

Find the model's implementation in vllm-project/vllm:
- Use `gh` CLI to find model files: `gh search code "repo:vllm-project/vllm {model_keyword}" --path vllm/model_executor/models`
- For VLMs, also search multimodal processing: `gh search code "repo:vllm-project/vllm {model_keyword}" --path vllm/multimodal`
- Read key implementation files to understand:
  - Model registration and configuration
  - Core class hierarchy (which vLLM base classes are extended)
  - Input processing pipeline (text + multimodal)
  - Forward pass implementation details
  - vLLM-specific optimizations applied
- See [references/model-research-guide.md](references/model-research-guide.md) for detailed code analysis methodology

**2e. Cross-Reference Documentation**

- Check vLLM docs: `https://docs.vllm.ai/en/latest/models/supported_models/`
- Search for known issues: `gh search issues "repo:vllm-project/vllm {model_keyword}" --label bug`

### Step 3: Generate the Tutorial Document

Read [references/style-guide.md](references/style-guide.md) for complete document structure and formatting conventions.

Key requirements:
- Write in **Chinese (简体中文)**, keeping English for technical terms
- Follow the multi-part structure defined in the style guide
- Include rich visual elements (see style guide for Mermaid diagram patterns)
- Include technical principle deep-dives for relevant mechanisms (MoE, MLA, GQA, ViT, etc.)
- Include document header with version info and date
- Include a "文档概述" section with target audience and reading guide

### Step 4: Save Output

Save the generated markdown file to `outputs/` directory relative to this skill's location:
- File path: `outputs/{model_name_snake_case}.md` (e.g., `outputs/qwen3_vl.md`, `outputs/deepseek_v3.md`)
- Create the `outputs/` directory if it doesn't exist
- Use snake_case for file names (lowercase, underscores)

The skill directory is the same directory as this SKILL.md file.

## Model Identification Heuristics

When the user's request is ambiguous, use these heuristics to determine what to include:

- If the model name contains "VL" or "Vision" → it's a VLM, include Part 5 (ViT computation)
- If the model is known MoE (DeepSeek-V3, Mixtral, Qwen3-MoE) → emphasize MoE in Part 2
- If the model has MLA (DeepSeek-V3 series) → include MLA deep-dive in Part 2
- If the selected checkpoint is dense + GQA (for example Qwen3 dense) → emphasize GQA analysis. Llama 4 uses MoE; inspect its router, shared/routed experts and checkpoint config rather than classifying it as dense.
- If the model generates images/video → check for DiT architecture and include if relevant

## Quality Checklist

Before saving the document, verify:
- [ ] Document has 4+ Mermaid diagrams (architecture, flow, sequence, class)
- [ ] Document has 3+ comparison/reference tables
- [ ] Document includes actual code snippets from vLLM source with file path annotations
- [ ] Document follows Chinese writing convention with English technical terms
- [ ] All sections have substantive content (no placeholder text)
- [ ] Document header includes version and date metadata
- [ ] Model series comparison table includes: model name, params, release date, key innovations, paper link, HF/ModelScope link
- [ ] Technical principle deep-dives present for relevant mechanisms
- [ ] Code location index table in appendix maps components to file paths
- [ ] For VLM models: Part 5 (ViT) is present and complete; For non-VLM: Part 5 is omitted

## References

- [references/style-guide.md](references/style-guide.md) — Document structure, formatting conventions, and output patterns
- [references/model-research-guide.md](references/model-research-guide.md) — Detailed research methodology for code analysis
- External: [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) — Comparative model architecture reference
