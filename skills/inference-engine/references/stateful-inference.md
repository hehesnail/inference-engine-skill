# 循环状态、混合模型与在线权重更新

用于 Mamba/RWKV、attention+SSM 混合模型、分块 prefill、prefix 复用、speculative rollback，以及训练与 rollout 共用推理引擎的场景。具体张量布局和支持范围以被测模型与引擎实现为准。

## 把一次请求的状态画清楚

| 状态 | 所有者及更新点 | 最容易漏的边界 |
|---|---|---|
| attention KV | cache pool、slot/block table、attention kernel | block sharing、滑窗、prefix 命中与 eviction |
| SSM/循环状态与卷积历史 | state pool、请求到 state slot 的映射 | 新请求清零、chunk 续跑、分支复制、取消后 slot 复用 |
| 位置与长度 | scheduler/runner/metadata | prefill 已处理长度、decode 增量、padding 与绝对 position |
| sampler/grammar/RNG | 请求级 sampling state | speculative 拒绝、重试和独立请求隔离 |
| adapter/权重身份 | model runner、权重管理器、cache key | 更新期间在途请求、旧 KV/循环状态、图捕获的地址 |
| 临时 workspace | kernel/runner | graph replay 的生命周期、跨 stream 读写和异步释放 |

纯 SSM 不存逐 token attention KV，但仍有固定维度循环状态、卷积历史和临时张量；混合模型还保留 attention 层的 KV。`d_state`、层数、head 组织、dtype 和状态分片决定容量，不能沿用单一 Mamba/RWKV 世代的公式。固定状态大小也不意味着能准确记住任意长历史。

对一个 prompt A 的三种执行建立对照：一次性 prefill；拆成 A1、A2 顺序执行；先执行无关 B，再复用其 slot 执行 A。前两者应在约定数值容差内一致；第三种不能受 B 污染。prefix 共享或分支生成时，循环状态需要明确快照与 copy-on-write 机制，不能把可变对象直接分给两个请求。

## 状态更新与输出路径

- prompt 整体 tokenize 后送入模型；逐字符 encode 再拼接不保证相同 token 序列。RWKV 一类递归 API 首次输入 prompt IDs，后续输入刚采样的 token IDs，不能以 `None` 代替新 token。
- logits/head、prompt logprobs、评分、pooling 可能复用 hidden states 或多次调用局部模块。读输出不能再次推进循环状态。先定位 runner 选取哪些 rows，再判断 forward 调用是否重复。
- chunk prefill 后保留状态、已处理长度和位置；连续 batching 重排时 state slot 与请求 ID 一起重排。结束、超时、取消和异常路径都要回收本请求所有状态。
- CUDA Graph 捕获时固定地址、真实 replay 时更新内容。以“第一次 replay 正确”验收，会遗漏 replay 仍读取旧输入或旧 state pointer 的问题。
- speculative 分支需要能回到已提交 token 的状态；只截断 KV 长度，不恢复循环状态/卷积历史/grammar，并不构成正确回滚。
- reset 后重复相同工作量才是可比较微基准；把状态初始化计时放在基线外、候选内会改变含义。

## 在线更新权重：一次事务，而非一次 copy

rollout 系统（slime、miles、verl）的有用部分是权重版本、推理缓存和训练进程之间的接口。训练配置、集群启动器和任务奖励不自动成为引擎工作流要求。

1. 明确版本 V 的模型、adapter、量化 scales、tokenizer/config 和 draft 身份。请求及训练消费的 rollout 记录版本，避免把跨版本 logprobs 当作同一策略。
2. 按引擎支持的入口建立更新会话与暂停/排空边界。确认 TP/PP/EP 各 rank、目标模型和 draft 的更新顺序；IPC 传句柄不免除 buffer 所有权、同步和存活期。
3. 更新返回失败时，判断是否已部分写入，而不是继续使用“旧版本”标签。通信 barrier 成功不等于所有权重、cache 和 draft 都一致。
4. 使旧 KV、prefix、SSM 状态失效，或由实现保证不同版本隔离；LoRA cache key 也需要正确区分 adapter。更新不能复用的 graph/量化 packed weights 需按实现重建。
5. 在所有相关组件完成后发布版本与恢复请求。用固定输入比较更新前后 logits/输出，并验证缓存命中路径和无缓存路径的一致性。只收到成功 HTTP 响应不足以证明接线正确。

**具体源码例子。** SGLang 固定快照 `8368f86f` 中，`scheduler_components/weight_updater.py` 的 tensor/distributed 更新要求先有 `begin_weight_update` 建立的会话。target runner 先更新后 draft 可能失败，代码仍安排必要的 cache flush；flush 遇到 running/queued requests 也可能失败。`engine.update_weights_from_tensor(..., flush_cache=True)` 的默认参数不能替代这个会话契约。迁移到其他版本时重新核对相应函数，勿把这里的函数名当作永久 API。

## 训练与推理数值对齐

训练与 rollout 的 mask、position、采样变换、logprob 定义、tokenizer、量化和 routing 需要逐项对应。MoE 的专家选择一致只覆盖 routing 的一部分，不能推导 bitwise logits 一致或 KL 必为零；FP8 的 scale 更新和重排顺序同样影响结果。LoRA 同步还要区分 adapter 增量、merged weights 和 fused/packed 缓存。

异步 rollout 与 partial rollout 允许吞吐优化，但需要记录生成片段实际使用的版本及重新计算 logprobs 的策略。不要仅凭“开启 async”认定训练消费的数据契约成立。OOM 先定位是 rollout KV、actor、reference、logprob 还是 optimizer 占用；降低训练梯度检查点内存不能保证解决 rollout 的 KV 问题。

按修改选择有效边界，不额外扩张已约定实验矩阵。与 [模型接线](model-integration.md)、[数值验收](numerical-validation.md)、[推测解码](speculative-decoding-contracts.md) 联用。

## 从状态契约进入源码

以下是在 vLLM `7b77ed581d92fa14e34ab43344837bb1ca1b4257` 核对的条件入口，使用前匹配实际 architecture 与版本。Mamba-1 与 Mamba-2 的状态形状/算子不同。

`模型注册/config → mamba_mixer2.MambaMixer2.conv_ssm_forward → Mamba2AttentionMetadataBuilder.build → BaseMambaAttentionMetadataBuilder → state slot / scan / final-state写回`。

- `vllm/model_executor/layers/mamba/mamba_mixer2.py`：从 forward context 取 metadata；`state_indices` 对应请求slot，`has_initial_states/prep_initial_states` 决定是否读取历史。
- `vllm/v1/attention/backends/mamba2_attn.py`：构建初始化状态与 checkpoint offsets。
- `vllm/v1/attention/backends/mamba_attn.py`：`_compute_chunk_metadata` 让 chunk 不跨请求并按绝对位置对齐；`_prefill_cpu_metadata` / `_compute_common_metadata` 连接 query length、已处理长度与 prefill/decode 分支。
- Mamba-1 另查 `vllm/model_executor/layers/mamba/mamba_mixer.py::MambaMixer.forward_impl`；不能直接套 Mamba-2 API。

先找第一个发生偏差的交接点。输入/初态相同而 chunk metadata 错，优先修 metadata；metadata 正确但漏读/错写slot，修生命周期；这些一致而局部算子先偏离，才进入 kernel。可复用测试定位包括 `tests/v1/attention/test_mamba_metadata_builder.py`、`tests/v1/worker/test_mamba_hybrid_model_state.py` 与 `tests/kernels/mamba/test_mamba_mixer2.py`。这里只定位，未将这些上游测试标作本次已执行。
