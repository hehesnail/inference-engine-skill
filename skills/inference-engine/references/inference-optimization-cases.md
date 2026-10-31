# 从 workload 到配置、量化与 speculative decoding

完整来源：gpu-perf-tune 32 个工作流、Orchestra 33 个推理相关专题、TensorRT-LLM 35 个专项；见 [工程目录](repository-guides.md)。以下是按需使用顺序，不是每次全部执行的矩阵。

## 建立能回答问题的 workload

定义输入/输出长度分布、prompt/tokenizer/chat template、到达过程、并发或QPS、cache状态、采样/stop、质量条件、资源和源码身份。记录实际 token 数、错误、取消和超时。固定并发与固定 request rate 问的是不同问题；报告 aggregate throughput、per-GPU throughput、goodput 与 latency 时标明各自分母。

gpu-perf-tune 的 evidence bundle、known-good config、baseline bridge/diff、value ledger 把配置、测量与成本归在一起；AA workload、fleet、Prometheus/DCGM/Zymtrace 只在当前服务与数据源存在时启用，时间锚点必须一致。没有 MCP/监控后端就用本任务日志/trace，不能把缺指标补成零。

capacity sizing 先分权重/KV/state/workspace，再核对实际分配日志。graph diff 和 decode step budget 结合 [Host/GPU 分析](host-device-analysis.md)：按设备/rank裁剪区间、求活动并集，保留 graph envelope 与真实 kernel busy 的区别。固定时间偏移、特定 NVTX 名和固定层数只属于来源例子。

## 选择优化方向

| 观察 | 应查内容 | 可区分原因的对照 |
|---|---|---|
| GPU 间有大空档、CPU 逐 token 工作多 | TRT host/sync/graphs，cfregly decode budget | 同一 workload 的 host wrapper、图模式、batch 与 launch；不要先写 GEMM |
| attention/KV 占主导 | prefix/chunk/cache策略、attention backend、布局 | 相同实际 token、cache初态和窗口；prefill/decode分别看 |
| MoE 尾部或通信突出 | routing分布、grouped GEMM、dispatch/combine、EP/NIC | per-rank真实routing与critical path；dummy均匀routing不能代替 |
| GEMM/归一化等单算子热 | NCU、生成码、fusion、kernel替换 | kernel+wrapper+整模型三级；明确copy、scale、padding成本 |
| 内存不足或权重流量占主导 | 量化/分片/offload/并发策略 | 容量、质量、加载成本与延迟同时记录 |
| speculative路径收益不稳 | draft/verify长度、acceptance、调度和cache | accepted tokens/step、draft+verify+rejection+host全部成本 |

## 量化

Orchestra 的 AWQ/GPTQ/HQQ/bitsandbytes/GGUF、TRT 与 HF 后端资料保留算法和使用细节，作为候选起点，接口以目标版本为准。权重格式、group size、对称性、zero point、scale布局、accumulation、激活/KV dtype、实际 kernel backend 要对上；文件变小不证明服务支持或更快。

校准数据代表目标分布，与评测集的作用分开；保存 checkpoint/tokenizer revision、校准参数和输出格式。不为使用某个上游 skill 自动量化当前模型。质量门槛沿用本任务约定，缺少质量测试时报告缺失；速度不能冲掉质量失败。排查 numerical drift 时比较相同输入、参考实现与中间层，避免两边都使用同一错误优化。

## Speculative decoding

将 drafter 训练/导出、服务接线、参数搜索分开。先核对 target/draft tokenizer、词表/embedding、位置与KV语义、采样/rejection实现；用路径证据和非零 acceptance 证明真的执行了 speculative 分支。

测量 draft时长、verify时长、每轮接受/拒绝长度、额外内存与调度开销；最终比较请求延迟/吞吐，而不是只报告 acceptance。改变并发、batch、输出长度或草稿长度会改变收益，保持对照条件。上游训练流程仅在任务需要新 drafter 时启用，不因研究 speculative 自动训练。

## TensorRT-LLM 的其他接入点

除此前 host 专题，本次补齐 Nsight Systems/Compute、workload、性能不稳/回归/casebook、Triton/CuTe/tile-IR、MoE、serve config、multimodal onboarding、FlashInfer upgrade、测试与 VisualGen。相应 engine 专属配置保持在完整指南中；跨引擎比较只对齐共同的 workload/计时边界，不强行把不同 scheduler/图模式参数解释成同一实现。

## 具体实现契约

[量化与稀疏](quantization-engine-contracts.md) 对齐算法、checkpoint、kernel和服务；[推测解码](speculative-decoding-contracts.md) 给出接受/残差采样及cache回滚；[attention](attention-backend-contracts.md) 检查位置、mask与layout。rollout/LoRA权重同步走 [状态与更新](stateful-inference.md)，不以复制完成代替cache和版本一致。
