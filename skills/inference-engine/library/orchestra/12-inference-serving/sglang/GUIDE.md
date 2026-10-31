# SGLang：服务、程序前端与缓存

SGLang 的 runtime 服务、OpenAI-compatible client 和 `@sg.function` 前端是不同接口。使用哪一层先明确，不把参数混传。正式接线、kernel、测试路由见 [SGLang 工作台](../../../../references/sglang.md)。

## 服务与客户端
命令基本形态为 `python -m sglang.launch_server --model-path "$MODEL" --host 127.0.0.1 --port "$PORT"`；可选 flags 查所用版本。Radix cache 常默认启用；比较时核对 `--disable-radix-cache`，不编造 enable flag。量化、speculative、TP、多节点配置均需模型/后端支持。

所核对的 `sg.OpenAI` 构造器首参数是 model_name，URL 通过支持的 client kwargs/后端配置传入。连接本地 runtime 使用对应 RuntimeEndpoint。`sg.gen` 支持 regex/json_schema，但无通用 tools/grammar 参数；`run_batch` 是相应 program 对象的方法，不是凭空拼出的顶层函数。

## 正确利用缓存与结构输出
prefix 按 token IDs 与实现中的附加身份匹配，不能只比较文本。冷缓存、热缓存、同时到达、块对齐、eviction、权重/adapter 更新和混合模型状态会改变可复用长度。减少多少 prefill tokens 不等于相同比例 TTFT 或端到端加速。

结构合法不等于业务正确。token budget、stop/cancel 和不支持的 schema keyword 仍可能失败；grammar state 需要与请求、speculative rollback 一致。看 [结构输出契约](../../../../references/tokenizer-and-structured-output.md)。

## 接线与部署
检查 scheduler、runner、cache pool、processor 与 kernel 实际路径。权重 tensor/distributed 更新存在会话与部分失败语义，见 [在线更新](../../../../references/stateful-inference.md)，不能只执行 copy 后继续复用旧 KV。多图与 diffusion 看 [专项路径](../../../../references/sglang-specialized-paths.md)。

本地配套 [部署](references/deployment.md)、[Radix](references/radix-attention.md)、[结构生成](references/structured-generation.md) 保留可用机制与边界，删去虚构 API、固定收益和无依据的缓存命中率门槛。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `12-inference-serving/sglang/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [deployment](references/deployment.md)
- [radix-attention](references/radix-attention.md)
- [structured-generation](references/structured-generation.md)

完整历史正文及本次具体勘误保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于追溯；运行规范使用本页与专题。
