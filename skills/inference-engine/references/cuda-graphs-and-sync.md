# CUDA Graph、host 同步与状态生命周期

若 eager 正确、图在换输入后仍产生旧结果，用同一 capture key 连续重放两个可区分的输入，分别与 eager 对照。检查第二次重放前输入与 metadata 的内容和地址，并在下一次重放前保存输出，区分读旧输入和输出被覆盖。若问题是 decode launch 开销、compile/fusion 收益或跨流等待，用 [Host/GPU 时间线](host-device-analysis.md)定位暴露在关键路径上的开销。

## Graph 接线检查

| 对象 | 必须保持的语义 | 常见失败 |
|---|---|---|
| 外部 input / metadata buffer | 被 capture 的地址与生命周期稳定；replay 前更新内容并建立 stream 顺序 | 给 Python 属性重新赋新 tensor，图仍读取旧地址 |
| Python scalar / launch config | capture 时固化；变化必须走支持的 tensor 参数、另一个 bucket 或 eager 路径 | 修改 temperature/长度变量却以为图中常量会自动更新 |
| 输出 / residual / scratch | 明确谁拥有、何时可复用；跨下一次 replay 消费时必要的复制和依赖仍存在 | 保存同一输出 view，后续 replay 覆盖前一请求结果 |
| KV page table、slot mapping、state 索引 | 数据可更新但布局、有效长度和 capture 假设匹配 | padding slot 写进真实请求，旧 page table 指向已释放页 |
| bucket / capture key | 包含改变执行与布局的条件；有效 token mask 正确 | 不同 speculative 步数、LoRA/多模态或后端误复用同一图 |
| stream / event | capture、replay 和调用方消费之间的依赖完整 | eager 下隐含同步遮住问题，graph 下暴露 race |
| custom op | dispatcher、mutates/alias、fake/meta 与真实 shape 一致 | compile 消掉副作用，或返回的 shape/dtype 与真实 kernel 不同 |

先读引擎现有 graph runner、capture size、piecewise boundary 和 fallback。新增旁路 graph 容易与调度、共享内存池、通信 capture、编译缓存和热重载冲突。

对 shape 分桶比较 padding 额外计算、图数量、显存与命中分布。小矩阵 decode、长上下文和 mixed prefill/decode 可能需要不同策略；不要照抄固定 buckets。只有问题涉及这些路径时才覆盖对应边界。

## 去同步不是删掉依赖

- `.item()`、Python 对 CUDA tensor 的 truth test、动态长度输出和 D2H 消费可能迫使 CPU 等待。沿真实调用链定位阻塞点；版本和算子实现会改变具体 CUDA API，不能仅凭名称下结论。
- GPU→CPU `non_blocking=True` 后，CPU 读取前仍需完成通知；提前读取可能得到旧值。CPU→GPU 异步复制也要保证源数据及 pinned buffer 在传输完成前不被修改或复用。
- `event.wait()` / stream wait 表达设备之间的先后，`event.synchronize()` 让 host 等待；是否能替换取决于谁消费结果。CUDA events 也有所属设备、stream 和生命周期。
- `torch.where` 三参数形式保留固定 shape，但两支可能都计算，且有副作用、非法输入、NaN/Inf 或额外算量时不能直接替换控制流。一参数形式与动态索引又是另一条路径。
- 静态 tensor、index、metadata 的复用，要规定更新时机、所有者和失效条件。request finish/cancel、batch shrink/grow、prefix reuse、recompute 后不能留下过期状态。
- 分布式 async collective 的 `work.wait()` 和 stream 关系依 backend、版本与环境而定。保留数据依赖，再看 timeline 是否发生 host 阻塞；不要笼统把所有 NCCL wait 当成 CPU synchronize。
- 为减少 Python 时间做 hoist/cache 时，确认被缓存的值是否随 batch、rank、模型切换、KV 长度变化。初始化快照不能代替运行时状态。

## Compile / fusion 对照

在匹配版本和输入下保存 A/B 的实际编译日志、FX/Inductor 代码或可用的 graph dump。日志开关与目录以安装版本为准，不依赖硬编码的 `=== FX GRAPH ===` 正则。

对照“源算子 → partition/graph break → lowered op → 实际 kernel/launch”的链路。节点更少不等于更快：可能发生额外 padding、layout conversion、spill 或 backend fallback。记录 graph hit/fallback 的样本范围，不把启动时 capture 成功当成所有请求都走图。

例：希望把 residual add 与 RMSNorm 融合。检查临时张量是否真的消失、原位输出是否保持、上下游是否继续使用旧 residual、是否仍进入同一 attention/backend。用 kernel/forward 对照回答局部收益，用请求测试回答 serving 收益；两者不可互相替代。

## 验证重点

围绕修改实际影响的条件，检查 eager 与 graph、capture 后输入/长度改变、同一 buffer 的重复 replay，以及受影响的 bucket 边界或 fallback。状态修改先核对算子结果和持久状态，再核对连续请求；不默认扩大成所有模型/硬件矩阵。

保存成功候选对应的源码、实际 launch 配置与必要证据，按 [优化工作流](optimization-workflow.md) 决定保留。TRT 的具体 pattern 与例子保存在 [Host 资料](../library/trtllm/host/GUIDE.md)；训练用 TE/Megatron wrapper 只在目标代码确实使用它们时参考。
