# vLLM 优化：先分解成本再选参数

固定源码/镜像、模型与tokenizer、实际输入输出IDs、并发/到达过程、cache初态、采样、失败处理和资源。初始化、warmup、稳态与端到端分别计时。

## KV、prefix 与调度

Paged allocation改善碎片与共享，并不保证“总显存减少50%”。普通attention KV原始量可从 layers×2×KVheads×head_dim×dtype_bytes×cached_tokens 估算，TP复制/分片、MLA、滑窗、混合SSM、padding/block和metadata按实现修正。70B是权重参数数量，不能直接推导KV160GB。对照启动分配日志。

例如三请求各有2000共同prefix+50新tokens，完全无复用处理6150个prefill tokens；理想复用下是2150。这个比值只反映待处理token数，不是2.86×实测加速；attention算量、queue、decode、cache查找与并发均改变时间。首请求、命中请求、eviction后请求分别记录。

chunked prefill平衡调度与延迟，不保证所有负载都快。调大max_num_seqs/max_num_batched_tokens会同时改变workspace、排队与实际batch，做一项有区分力的配对。

## 量化、图与推测

量化先验收质量与实际kernel，见本目录quantization。图模式区分capture成本、CPU launch与kernel执行，replay需更新真实输入；eager更快可能来自图shape/padding等，不仅是host问题。

target与draft需要tokenizer/词表、位置、采样与缓存协议兼容。Llama3 target和任意TinyLlama draft不能仅按大小直接搭配。标准精确接受算法不以“容忍2–3%质量下降”作为固有代价；近似方案则明确质量契约。

## 正确基准入口

固定源码中throughput/latency使用`--input-len`、`--output-len`等参数，而非旧文`--input-tokens`。不同版本random dataset参数可能迁移，先查CLI help。throughput离线总速率不能给出在线TTFT/ITL；serve客户端要记录每请求时间与错误。保存退出码（bash管道用pipefail），实际tokens和成功/失败数，不只接受漂亮的峰值吞吐。
