> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[BBuf/AI-Infra-Auto-Driven-SKILLS@c592955044e8](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/blob/c592955044e896edd20890468c9bdbbe30ab0be6/skills/model-architecture-diagram/SKILL.md)，2026-10-31 核对。

# Model Architecture Diagram

## Workflow

Return only public original diagrams indexed by this skill.

1. Run the bundled resolver:

```bash
python3 skills/model-architecture-diagram/scripts/model_architecture_diagram.py "<model name>"
```

2. If the resolver returns `kind: existing`, return the raw image Markdown it prints and preserve the source attribution line.
3. If the resolver returns `kind: no_match`, tell the user that no public original architecture diagram is indexed for that model.

## Source Priority

Use `references/diagram-index.json` as the source of truth. It stores raw GitHub image URLs and official vendor/CDN or pinned Hugging Face
`resolve/<sha>` URLs. Official sources include Qwen and XiaomiMiMo; GitHub sources:

- `datawhalechina/self-llm`
- `CalvinXKY/InfraTech`
- `Tongyi-MAI/Z-Image`
- `Wan-Video/Wan2.1`
- `Wan-Video/Wan2.2`
- `Tencent-Hunyuan/HunyuanVideo`
- `Tencent-Hunyuan/Hunyuan3D-2`
- `brayevalerien/Flux.1-Architecture-Diagram`

Prefer detailed implementation, cookbook, or architecture-card diagrams over
paper figures. Good sources show module boundaries, dataflow, MoE / attention /
cache paths, or model-specific runtime structure rather than only a high-level
paper overview. Official repository diagrams and curated implementation
diagrams are first choice; paper figures are fallback only when no more detailed
public original diagram is indexed.

Do not copy remote image binaries into the skill. Return the indexed original URLs so the chat renderer can display the original image.

## Existing Diagram Rule

For a direct match, show the original image. Good direct matches include:

- DeepSeek V3/V3.2/V4, GLM-5, Kimi K2/K2.5/K3, MiniMax M2.5, Qwen3.5, Qwen3-VL, and Step 3.5 Flash from InfraTech.
- Qwen3.8-Flash-Next has an official diagram (also named `qwen4_exp` in runtime code). Plain Qwen3.8 2.4T remains `no_match`; these are different architectures.
- MiMo V2.6, MiMo V2.5 Pro and GLM-5.2 have indexed originals. GLM-5.3 remains `no_match` because no direct diagram was verified.
- Hunyuan-A13B, Kimi-VL, Qwen3, Qwen3-VL detail flows, MiniMax M2, and Llama 4 architecture/module diagrams from self-llm.
- Z-Image, Wan2.1, Wan2.2, HunyuanVideo, Hunyuan3D 2.0, and FLUX.1 diffusion architecture/module diagrams from public GitHub sources.

If multiple diagrams match, show all high-confidence matches up to the resolver's default limit. For example, DeepSeek V3 may return the full architecture plus MLA MHA/MQA diagrams.

## Hosted Original Diagram Gallery

Do not commit the `sgl-cookbook-model-architecture-images/` gallery into the repository. The public-original image set is hosted as a GitHub Release asset and indexed by a GitHub issue.

Current hosted artifact:

- Issue index: https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/issues/31
- Release page: https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/releases/tag/sgl-cookbook-architecture-images-2026-05-02
- Zip download: https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/releases/download/sgl-cookbook-architecture-images-2026-05-02/sgl-cookbook-model-architecture-images-2026-05-02.zip
- Digest: `sha256:ea432081849a250429d3d1ecf246e267c5cc42f989aaf4b9ca695b581e7fa50f`

The gallery is a 2026-05-02 snapshot containing 44 public original diagram image files; it does not mirror the current index, plus a lightweight `index.html`, `index.md`, `manifest.json`, HTML contact sheet, and `architecture-audit.md`.

To inspect the gallery locally:

```bash
curl -L -o /tmp/sgl-cookbook-model-architecture-images-2026-05-02.zip \
  https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/releases/download/sgl-cookbook-architecture-images-2026-05-02/sgl-cookbook-model-architecture-images-2026-05-02.zip
unzip -q /tmp/sgl-cookbook-model-architecture-images-2026-05-02.zip -d /tmp
open /tmp/sgl-cookbook-model-architecture-images/index.html
```

## Useful Commands

List known original diagram aliases:

```bash
python3 skills/model-architecture-diagram/scripts/model_architecture_diagram.py --list-known
```

Emit JSON for automation:

```bash
python3 skills/model-architecture-diagram/scripts/model_architecture_diagram.py "GLM-5" --format json
```

## References

- `references/diagram-index.json`: original diagram link index and aliases.
- `references/source-notes.md`: audited source repositories and local cache paths.
