# Whisper：音频窗口、任务与真实计时

固定音频采样率、声道、重采样、语言、task、VAD、模型与解码设置。转写与翻译的支持不同；turbo以转写为主，翻译应选相应多语言模型与translate任务，不能用模型名替代任务配置。

## 分段与执行
模型窗口化处理长音频，不等于任意实时服务协议。跨窗口保留的上下文、重叠、时间戳和去重影响文本；长音频是否分段由实现与任务决定，“超过30分钟必须…”不是通用限制。

faster-whisper的segments为惰性迭代，计时需覆盖实际遍历；返回generator不代表在线流式服务。循环每个文件执行transcribe也不是batch。若需要streaming，明确音频到达、窗口提交、partial/final和重写文本规则。

## 测量
RTF=墙钟秒/音频秒，说明加载、I/O、预处理、VAD、encoder、decoder、batch和并发是否包含。WER/CER固定文本规范化；翻译质量另选合适评价，不将转写分数直接挪用。参见 [多模态推理](../../../../references/multimodal-inference.md)。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `18-multimodal/whisper/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [languages](references/languages.md)

完整历史正文及本次具体勘误保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于追溯；运行规范使用本页与专题。
