# Attention 后端、位置与长上下文

用于 attention 替换、FlashAttention/SDPA 接线、RoPE/ALiBi、长上下文和 decode 不一致。来源是 Orchestra 的 FlashAttention/long-context 指南，经实际引擎与官方接口核对后保留以下契约；不是要求每次遍历全部后端。

## 把一次调用说清楚

记录 Q/K/V 的逻辑 shape 与物理 stride、Q heads/KV heads/head dim、query/key 长度、序列/请求边界、position IDs、cache slot/page 表、scale、dtype、mask/window/ALiBi 和输出布局。相同 tensor 大小不代表相同 attention。

- PyTorch SDPA 常用 `[B,H,S,D]`；独立 `flash_attn_func` 用 `[B,S,H,D]`。转换须包含输出转回，计时说明是否包含 transpose、contiguous/copy。
- GQA/MQA 需要正确的 Q-head 到 KV-head 映射与可整除关系；“K/V heads 更少”不证明任意后端会自动接受。SDPA 的相关参数、内核与设备支持随版本变化。
- 变长 prefill 用各序列的长度/累积长度，不能把 packed tokens 当成一条长序列。跨请求注意力必须被隔离。
- `dropout_p` 在推理时取零；SDPA 不会仅因外层 module.eval() 自动把显式非零 dropout 改为零。
- FlashAttention 是精确 attention 算法的一种 IO 实现，浮点执行顺序仍会导致差异；容差按模型、dtype 和参考运算确定，不能固定“fp16 最大误差 <1e-3”。

## 矩形 causal mask：最容易把 prefill 对、decode 错的接线

当 query 长度 2、KV 长度 5，末尾两 token 的 causal mask 应为：

~~~text
1 1 1 1 0
1 1 1 1 1
~~~

FlashAttention 2.1 改为右下对齐。SDPA `is_causal=True` 的非方阵语义不能默认与它相同；核对当前接口，必要时传显式正确 mask。单 token decode 若已经只有过去及当前 KV，不能误用左上三角让它只看到第一个 key。核对 zero-length、全 mask 行、滑窗边界、padding 和 prefix/chunk 的实际位置。

[FlashAttention 版本说明](https://github.com/Dao-AILab/flash-attention#21-change-behavior-of-causal-flag)、
[PyTorch SDPA 接口](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)。

## RoPE / ALiBi 接入

RoPE 的 interleaved 与 split-half 配对方式、rotary dimension、base、缩放和 attention scale 必须与权重训练时一致。不能只把 sin/cos 数值对上而忽略元素配对。cache 中保存旋转前还是旋转后的 key，也决定后续是否会重复旋转。

已有 128 个 prefix token，再 prefill 一个 64-token chunk，positions 应对应 128…191；下一次 decode 从 192 开始。不能每次 forward 重置为 arange(chunk_len)。请求重排时 positions、seq lengths、slot map 和 recurrent state 同步重排。

Orchestra 的示例在 CPU buffer 上调用 type_as，会把新建的 CUDA positions 转回 CPU。实现应显式把频率放在输入设备，并在适当高精度中算角度，再按目标接口输出 dtype。分块与一次 prefill 的位置必须一致。动态缩放改变频率时还要核对旧 key 的语义，不能只扩大 sin/cos 表。

修改 model.config 不保证已构造的 rotary module 重建；在实例化前准备匹配的 config，并从 checkpoint 加载。AutoModel.from_config 创建随机权重，不是加载预训练权重。max_model_len 是运行容量限制，不会赋予模型训练外的长上下文能力；文本重复次数也不等于 token 数。

ALiBi 是 attention bias，不替代 causal mask。causal 情况下可用 key_position-query_position 的有符号距离；其他 mask 语义不能直接套这个公式。ALiBi 不意味着无限可用上下文。显式构造 `S×S` bias 的教学代码可能抵消后端节省的内存。

## 证明后端真实生效

PyTorch SDPA、flash-attn 2、Hopper 的 FlashAttention 3 与 CuTeDSL/其他实现是不同接口/构建目标。安装 flash-attn 后把输入 cast 成 FP8，不证明已运行 FA3。FP8 scale/descale、tensor layout 与支持的 GPU 以该实现为准；Turing 支持也不能由 FA1、第三方移植推广为 FA2 主实现支持。

对照先复用一份输入和正确 mask，以可信参考计算输出；再预热实际 dispatch，移出 RNG、加载、编译与不对称的分配。必要时用当前 PyTorch `torch.nn.attention.sdpa_kernel` 限定后端，检查不支持时的错误与 profiler 的实际 kernel；普通自动 SDPA 调用可能回退。性能结果分别报告 kernel、布局转换/封装、模型或请求成本。

[官方安装与硬件矩阵](https://github.com/Dao-AILab/flash-attention#installation-and-features)。进一步查 [kernel 接线](kernel-engine-integration.md)、[数值验证](numerical-validation.md)、[状态生命周期](stateful-inference.md)。
