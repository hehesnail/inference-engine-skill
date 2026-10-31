# Hugging Face Tokenizers 在推理链中的使用

重点是 checkpoint 与输入/输出契约一致，训练新 tokenizer 仅在任务明确需要时进行。

## 组成与一致性
normalizer、pretokenizer、model（BPE/WordPiece/Unigram）、postprocessor、decoder 分别检查。ByteLevel pretokenizer 要配 decoder；BERT 输入通常还要 CLS/SEP postprocessor。自定义 special tokens 的插入顺序决定 ID，不能将“PAD=3”等新词表约定套到 BERT 等 pretrained 模型。

固定 tokenizer 文件/revision、added tokens、chat template、BOS/EOS/PAD、padding side、truncation 与是否添加 special tokens。批量 encode 的 attention mask 与有效长度一起交给模型；EOS 兼任 PAD 时不能按 ID 值删除所有 EOS。

## 检查方法
使用实际被编码文本检查 offsets、空格、Unicode/emoji、空字符串与模板边界；normalization 可能不可逆，不承诺字符串逐字往返。模型需要 fast tokenizer 的路径确认实际 fast/slow 实例与回退行为。整段 encode 不等于逐字符 encode 再拼接。

训练 tokenizer 后除保存 JSON，也保存相应配置及 special-token/template 定义；只替换 tokenizer.json 可能改变模型词表映射。多进程批量处理按平台用主入口保护，不能把进程启动或下载时间混作 tokenize 速度。

[Tokenizer 与结构输出契约](../../../../references/tokenizer-and-structured-output.md) 连接服务、grammar 和增量 detokenize 的边界。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `02-tokenization/huggingface-tokenizers/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [algorithms](references/algorithms.md)
- [integration](references/integration.md)
- [pipeline](references/pipeline.md)
- [training](references/training.md)

完整历史正文及本次具体勘误保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于追溯；运行规范使用本页与专题。
