> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[shen-shanshan/vllm-dev-skills@1ccafeebf71b](https://github.com/shen-shanshan/vllm-dev-skills/blob/1ccafeebf71b5d922f351a0cd009a258b7474fac/skills/vllm-feature-tutorial/SKILL.md)，2026-10-31 核对。

# vLLM Feature Tutorial Generator

Generate a comprehensive code walkthrough tutorial document for a given vLLM feature or module.

## Workflow

### Step 1: Identify the Feature

Extract the feature/module name from the user's request. Examples:
- "spec decode" / "speculative decoding"
- "chunked prefill"
- "automatic prefix caching"
- "tensor parallelism"
- "KV cache management"
- "continuous batching"
- "LoRA"
- "multimodal"

### Step 2: Research the Feature

Gather information from multiple sources. This is the most important step — thorough research determines document quality.

**2a. vLLM Official Documentation**
- Fetch relevant pages from `https://docs.vllm.ai/en/latest/` using WebFetch
- Look for design docs, API references, usage guides

**2b. vLLM Source Code**
- Use `gh api` or WebFetch to browse the vLLM GitHub repo (`vllm-project/vllm`)
- Identify core source files for the feature (use GitHub code search or browse directory structure)
- Read key implementation files to understand:
  - Core classes and their responsibilities
  - Key interfaces and method signatures
  - Data flow and control flow
  - Important algorithms and data structures

**2c. Related Resources**
- Search for relevant blog posts, papers, or design documents
- Check vLLM GitHub discussions/issues for design rationale

### Step 3: Generate the Tutorial Document

Read [references/style-guide.md](references/style-guide.md) for the complete document structure and formatting conventions.

Key requirements:
- Write in **Chinese (简体中文)**, keeping English for technical terms
- Follow the multi-part structure defined in the style guide
- Include rich visual elements:
  - **Mermaid flowcharts** for workflows and data flow
  - **Mermaid architecture diagrams** (flowchart TB with subgraph) for system overview
  - **Mermaid sequence diagrams** for component interactions
  - **Mermaid class diagrams** for class hierarchies
  - **Tables** for parameter references, method comparisons, performance metrics
  - **Code snippets** with file path annotations from actual vLLM source
  - **LaTeX formulas** for algorithm analysis (where applicable)
- Include a document header with version info and date
- Include a 文档概述 section with target audience and reading guide
- End with appendices: code location index and glossary

### Step 4: Save Output

Save the generated markdown file to `outputs/` directory relative to this skill's location:
- File path: `outputs/{feature_name}.md` (e.g., `outputs/spec_decode.md`, `outputs/chunked_prefill.md`)
- Create the `outputs/` directory if it doesn't exist
- Use snake_case for file names

The skill directory is located at the same directory as this SKILL.md file.

## Content Depth Guidelines

- **Part 1 (Basics & Architecture)**: Start from first principles. Explain the "what" and "why". Include theoretical analysis with formulas if the feature involves algorithms. Provide architecture overview with Mermaid diagrams.
- **Part 2 (Core Interfaces)**: Analyze base classes, interfaces, factory methods. Include class diagrams and method signatures.
- **Part 3 (Deep Implementation)**: Walk through actual vLLM code. Include code snippets with annotations. Trace execution flow with concrete examples.
- **Part 4+ (Variants/Comparisons)**: If the feature has multiple implementations or methods, compare them in tables.
- **Final Part (Configuration)**: Provide practical usage guidance, key parameters, tuning tips.
- **Appendix**: Code location index table mapping components to file paths.

## Quality Checklist

Before saving the document, verify:
- [ ] Document has 3+ Mermaid diagrams (architecture, flow, sequence/class)
- [ ] Document has 3+ comparison/reference tables
- [ ] Document includes actual code snippets from vLLM source with file path annotations
- [ ] Document follows the Chinese writing convention with English technical terms
- [ ] All sections have substantive content (no placeholder text)
- [ ] Document header includes version and date metadata

## References

- [references/style-guide.md](references/style-guide.md) — Document structure, formatting conventions, and content patterns
