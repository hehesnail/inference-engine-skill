# Tokenizer、模板与约束输出的引擎契约

这部分用于请求进入引擎前后的行为：文本到 IDs、模板与 special tokens、增量 detokenize、grammar/logit mask、stop 与重试。结构合法、业务合法与生成成功是不同检查。

## 文本到 token 的身份

固定 tokenizer revision、normalizer、added tokens、chat template、BOS/EOS/PAD、工具 schema、是否添加 special tokens，以及截断/最大长度。同一自然语言问题经不同模板得到不同 token IDs，就不是严格等价的性能或质量输入。

- BERT 的 `[PAD]` 常为 0；新建词表中按顺序指定的 ID 不自动兼容 pretrained checkpoint。CLS/SEP 等 postprocessor 也需与训练一致。
- ByteLevel pretokenizer 与对应 decoder 配对，验证空格、非 ASCII、emoji 和字节回退。SentencePiece 的默认 normalization/额外空白折叠可能改变输入，不能宣称任意文本逐字往返。
- SentencePiece 的 `character_coverage` 通常对大字符集语言留少量长尾、小字符集可为 1.0；语料与实际默认值决定选择。BOS/EOS/PAD/UNK 必须区分模型的有效 ID，不能把 T5 的 EOS 值同时当 BOS。
- offset mapping 是相对于 tokenizer 所定义的原始/规范化输入坐标；用被编码的那份文本验证切片，不能拿另一段字符串检查。
- 整段 encode 与逐字符 encode 再拼接不同。对已经模板化并 tokenize 的输入，避免二次添加 BOS 或 chat special tokens。
- fast tokenizer 不可用时是否回退到 slow、截断方向、padding side 和 attention mask 都要记录。padding token 与 EOS 共享时，也不能仅按 token 值推断有效长度。

用少量真实边界样本检查输入 IDs、mask、可见输出和 stop：空串/空白、Unicode、多轮模板、工具调用、截断边界。按任务选择，不把完整 tokenizer 训练流程引入普通推理排障。

## 结构约束进入了哪一层

| 机制 | 实际保证 | 额外成本与边界 |
|---|---|---|
| 请求侧 JSON/schema 参数 | 服务接受的 schema 子集与所用后端 | 不支持的 keyword、编译失败、冷编译/复用 |
| token 级 grammar/logit mask | 每一步允许的 token 集合 | grammar state、tokenizer 映射、dead end、EOS |
| 输出后 Pydantic 校验 | 已完成文本的解析与业务验证 | 无法倒推之前每步都受约束；可能触发重试 |
| 提示词要求格式 | 模型倾向遵循 | 不构成形式保证 |

CFG 不能普遍等同有限状态机；递归结构需要相应 parser/automaton。JSON Schema 的语法约束也不等于任意 Python validator、日期合法性、字段间关系或事实正确性。正则 `\d{4}-\d{2}-\d{2}` 可接受不存在的日期；是否允许尾随文本取决于 match/fullmatch 与锚点。

后端的能力必须由实际 API 和实现确认。例如核对过的 SGLang `sg.gen` 签名有 `regex`、`json_schema`，不支持任意 `tools=` 或 `grammar=` 关键字；工具调用应走当前支持的接口。Guidance 的本地 token 控制能力不能直接推广到远程 OpenAI/Anthropic backend；其 `gen` 并无通用 `grammar="<tag>..."` 参数。Outlines 新旧 API 差异较大，按安装版本选择模型适配器、Generator 和 schema 类型，不把旧 `generate.json` 示例拼接到新 API。

## 请求状态与终止

grammar 编译可以复用，但每个请求的解析位置、工具状态与输出 buffer 必须隔离；speculative accept/reject 需要提交/回滚相同状态。cancel、timeout、max_tokens、stop、EOS、服务错误都可能造成不完整 JSON，不能宣称“100% 有效输出”而只统计完成样本。

streaming partial object 的字段可能尚为 `None`，只有最终校验成功才能当成完整对象。SSE chunk 的边界不是 JSON/token/Unicode 边界；聚合解码时保留残片。重试会产生额外请求与 tokens，报告首次成功率、最终成功率、重试数及端到端延迟，不只展示最后一次成功。

Instructor 的异常取决于 provider 与 wrapper；重试耗尽不总是裸 `ValidationError`。日期、范围或 null 策略由业务契约决定，不用 fabricated default 掩盖缺失。工具执行由明确白名单与参数校验连接，不能对模型返回的字符串直接 `eval`。

与 [离线评测](offline-evaluation.md)、[数值验收](numerical-validation.md)、[推测解码状态](speculative-decoding-contracts.md) 配合。上游详细例子位于 Orchestra 的 tokenization 与 prompt-engineering 子目录；接口与保证范围以这里的分层和被测版本为准。

## SGLang grammar 到 streaming 的源码路线

固定 SGLang `8368f86fd7a71b07e2b7dda5d4146695ec54cbfa` 的 Python OpenAI 路径为：

`serving_chat/protocol → GrammarManager → backend matcher → scheduler grammar barrier → speculative mask → batch result提交 → detokenizer → SSE`。

- `python/sglang/srt/entrypoints/openai/protocol.py::ChatCompletionRequest.to_sampling_params` 与 `serving_chat.py::_convert_to_internal_request`：确认 response_format 真正转换为支持的约束；仅看到请求字段不证明 mask 已生效。
- `python/sglang/srt/constrained/grammar_manager.py::GrammarManager.process_req_with_grammar`：建立请求grammar。`base_grammar_backend.py::get_cached_or_future_value` 在命中后copy；`xgrammar_backend.py::XGrammarGrammar.copy` 复用compiled context但新建matcher。
- `python/sglang/srt/speculative/spec_utils.py::traverse_tree/generate_token_bitmask/build_grammar_vocab_mask`：候选树的临时accept/rollback与请求行mask对齐。
- `python/sglang/srt/managers/scheduler.py::_advance_pending_grammar` 及 `scheduler_components/batch_result_processor.py::advance_grammar_fsm`：上一批已提交token推进后才构造下一批mask；`grammar_advanced` 防止重复推进，`_resolve_spec_v2_tokens` 使用保留前缀并调整提交长度。
- `python/sglang/srt/managers/detokenizer_manager.py`：rid对应增量状态，完成后的回收与迟到输出需核对；`serving_chat.py::_handle_streaming_request/_generate_chat_stream` 连接SSE和最终状态。

若committed IDs已违约，先查mask、树回滚、barrier与重复推进；IDs合法而文字被截，沿detokenizer的rid/offset/stop与SSE客户端聚合继续查。length/abort/timeout造成的不完整JSON不能直接归因grammar错误。该快照已实现这些保护，不据假设症状宣称当前源码缺失它们。

相关测试入口为 `test/registered/unit/managers/test_batch_result_processor_spec_grammar.py`、`test_grammar_stop_speculative.py`、`test_incremental_detokenize.py` 及 `test/registered/spec/eagle/test_eagle_constrained_decoding.py`。仅定位未执行；其他frontend/backend按实际实现重新定位。
