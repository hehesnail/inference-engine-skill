# SentencePiece：模型、normalization 与 special tokens

SentencePiece 可训练 Unigram/BPE 等子词模型。推理排障先使用模型原配 tokenizer，不因压缩率或示例词表大小而重训。

## 训练与编码约束
记录 model_type、vocab_size、语料抽样与 normalization。`character_coverage` 对大字符集通常允许极少长尾落入其他处理，小字符集可为 1.0；不能反过来把 CJK 一律设 1.0、小字母表一律截断。参考选项默认值不代替目标语料设计。

默认 nmt_nfkc 与额外空白处理可能改变输入；需要保真时验证具体规则。BOS/EOS/PAD/UNK IDs 由 tokenizer 和模型配置共同决定；不能把 T5 EOS=1 同时用作 BOS。SentencePiece 默认 IDs 与 HF 包装层的约定可能不同。

抽样 encode 要明确 enable_sampling、nbest_size、alpha；确定性评测通常禁用随机分词。byte fallback、未知字和 normalizer 的行为用实际样本验证。训练语料与模型性能评测集分开。

## 引擎接线
比较 text→IDs→模型→IDs→text 的完整链，尤其是 chat template、特殊 token 重复、空格边界和流式不完整字节。多请求使用相同模型文件，缓存 key 依据真实 token 序列。性能结论必须有输入长度与批处理条件，不使用指南中的固定百万 token/s 表作为当前结果。
参见 [Tokenizer 契约](../../../../references/tokenizer-and-structured-output.md)。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `02-tokenization/sentencepiece/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [algorithms](references/algorithms.md)
- [training](references/training.md)

完整历史正文及本次具体勘误保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于追溯；运行规范使用本页与专题。
