# Speculative decoding：正确性、状态与收益

用于 draft/verify 接线、接受率异常、输出差异与性能对照。来源中的算法概念可用；Medusa、EAGLE/MTP、ngram、传统 draft model 的训练要求和接受规则分别核对，不能混成一个接口。

## 采样分布不能省略 rejection correction

传统精确 speculative sampling 中，目标分布 p 和草稿分布 q 必须是对同一前缀、同一 token 空间及实际采样变换后的分布。抽取 y~q 后，以 min(1,p(y)/q(y)) 接受。拒绝时从归一化的 max(p-q,0) 重采样；不是直接从 p 重采样。全部 K 个草稿通过时还可以从相应目标分布采样一个 bonus token，具体执行与目标方法契约一致。

两 token 例：p=(0.8,0.2)，q=(0.2,0.8)。接受概率质量为(0.2,0.2)，剩余0.6全部应给第一个 token。拒绝后直接采 p 会得到(0.68,0.32)，不再是 p。原来源的简略循环只返回 accepted token，不能当作正确采样器实现。

Greedy 模式按目标选择验证，与概率接受模式分开。温度、top-p/top-k、penalties、grammar mask 等发生在什么阶段会改变 p/q；不能用原始 softmax 做比率、另一边却按截断分布采样。多 token 约束需随已接受前缀推进，不能一次使用旧 grammar state。

论文/实现入口：[Speculative sampling](https://arxiv.org/abs/2211.17192)、
[vLLM speculative config](https://github.com/vllm-project/vllm/blob/7b77ed581d92fa14e34ab43344837bb1ca1b4257/vllm/config/speculative.py)。
Medusa 的 typical acceptance 和调过的 base model 不自动保持原始 target 分布；“所有方法零质量损失”应删除。不同 tokenizer 的 assisted decoding 需要显式映射机制，不能按 token ID 直接比较。

## 从一轮 token 走到状态

记录草稿前缀、候选长度、target 验证的行/position、接受长度、补充 token、最终提交 token 数。拒绝后 target 与 draft 的 KV、recurrent state、位置计数、block/slot、grammar状态必须回到已提交前缀，再接新 token。全接受、首 token 拒绝、部分接受、EOS/stop、取消、到 max length 是不同状态转移，只覆盖本次修改会影响的转移。

Tree verification 的 attention mask 应限制在每个候选自己的祖先；不能让兄弟分支互相看见。验完的 tree buffer 不等于所有节点都能提交到 cache。EAGLE/MTP 还要对齐 hidden-state层、词表映射和 draft checkpoint 配置；某个 GLM 模型的辅助层编号不推广到其他模型。

## 版本与可执行入口

当前核对的 vLLM 使用 `speculative_config` / `--speculative-config`，不使用旧教程的 use_v2_block_manager、speculative_model 顶层组合。SGLang 的 algorithm、draft_model_path、steps/topk/draft_tokens 是分别声明的配置，模型/方法可能有额外限制。先用本次 checkout 的 example/help 构造配置，不为使用 speculative 自动替换现有引擎或训练新 drafter。

小模型参数量比例不是成本模型：较小 draft 也可能因词表、带宽、并行通信或低接受率抵消收益。直接把 Medusa wrapper 填进 Transformers assistant_model 也不证明接口兼容。

## 该测什么

按相同 target、workload、输出约束、资源和 cache 初态比较。记录 proposed、accepted、committed tokens 的不同分母，以及每轮 draft、target verify、rejection/状态回滚、host 和调度成本。高 acceptance 不等于高吞吐；verify 更宽会占算力与内存，持续批处理的最优设置可能与单请求相反。

通过分支日志/trace 和实际非零草稿工作证明启用；仅有配置字段不够。Greedy 用匹配策略的输出/状态对照；概率路径用实现和分布契约及本任务指定质量证据，不能机械要求两种随机算法同 seed 逐 token 相等。正式性能测量移除 profiler，更多内容见 [benchmarking](benchmarking.md) 和 [数值验证](numerical-validation.md)。

SGLang的真实grammar barrier、树mask、结果保留/提交和流式输出定位见 [源码路线](tokenizer-and-structured-output.md)。这些保护是否存在及是否被执行需按被测版本分别确认。
