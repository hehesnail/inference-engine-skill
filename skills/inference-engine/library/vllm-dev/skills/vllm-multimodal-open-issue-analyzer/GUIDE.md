> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[shen-shanshan/vllm-dev-skills@1ccafeebf71b](https://github.com/shen-shanshan/vllm-dev-skills/blob/1ccafeebf71b5d922f351a0cd009a258b7474fac/skills/vllm-multimodal-open-issue-analyzer/SKILL.md)，2026-10-31 核对。

# vllm 多模态 Open Issues 分析

## 工作流

1. 运行 `scripts/fetch_multimodal_issues.py`
2. 脚本自动抓取、去重、分类，生成 Markdown 报告
3. 报告默认保存至 `<skill_root>/outputs/vllm_multimodal_issues_<timestamp>.md`
4. 告知用户报告路径及 issue 总数

## 运行脚本

```bash
python3 scripts/fetch_multimodal_issues.py
```

可选参数：

| 参数 | 说明 | 示例 |
|------|------|------|
| `--token` | GitHub token（有 gh CLI 时不需要） | `--token ghp_xxx` |
| `--output-dir` | 自定义报告输出目录 | `--output-dir /tmp/reports` |
| `--keywords` | 追加额外关键词（逗号分隔） | `--keywords "DeepSeek-VL,Kimi"` |
| `--stdout` | 输出到终端而非文件 | `--stdout` |

## 搜索关键词覆盖范围

**通用术语**: multimodal, multi-modal, VLM, vision language, image input

**VL 模型**: Qwen-VL/2/2.5, LLaVA, InternVL, MiniCPM-V, Phi-3/4-vision, PaliGemma, Florence, Pixtral, CogVLM, LLaMA-Vision, Idefics

**Omni/音频**: Qwen-Omni/2/2.5, audio, whisper, speech

**视频**: video understanding, video input, temporal

**特性关键词**: ViT, visual encoder, cuda graph, EPD, prefix caching, image token, image patch

## 问题类型分类逻辑

脚本对每个 issue 的标题、label、body（前 500 字符）做关键词匹配，归入以下类别（一个 issue 可属于多个类别）：

- Bug / 错误报告
- Feature Request / 功能请求
- Performance / 性能问题
- CUDA Graph / 计算图
- EPD / 编解码分离
- Prefix Caching / 前缀缓存
- ViT / 视觉编码器
- Video Multimodal / 视频多模态
- Audio / Speech / 音频语音
- VL Model: Qwen-VL系列 / Qwen-Omni系列 / LLaVA系列 / InternVL系列 / 其他模型
- Image Token / 输入处理
- Compatibility / 兼容性
- Documentation / 文档
- 其他 / Other

## 报告结构

生成的 Markdown 报告包含三部分：

1. **Issue 总览**（表格）：编号、标题、作者、创建日期、Labels
2. **按问题类型分类**：每个类别列出匹配 issue（含链接、创建日期）
3. **分类统计**：各类别 issue 数量汇总表

## 认证说明

- **有 gh CLI**（推荐）：复用已认证会话；GitHub 的主/次级速率限制仍适用。遇限流记录响应并遵循 retry/reset 信息。
- **无 gh CLI，有 token**：使用 `--token`
- **无认证**：搜索批次间自动等待 6 秒，避免触发 GitHub 10 req/min 限制
