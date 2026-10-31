# Radix prefix cache：复用什么、何时可复用

Radix树组织已处理token前缀及其cache引用。文本示意帮助理解，但实际匹配依token IDs、块/页对齐及实现中的附加身份。vLLM等引擎也支持prefix cache，不能拿“vLLM无缓存”作比较前提。

## 机制与生命周期
插入和查找沿最长已缓存前缀；分叉共享不可变prefix，后续写入需要正确引用/所有权。活跃节点受锁或引用保护，eviction策略只能回收可释放状态。具体LRU/FIFO等名称查实际`radix_eviction_policy`，不套旧`--eviction-policy`。

权重/LoRA版本、multi-modal processor与混合循环状态会影响cache可复用性。更新权重后旧KV并不仍有效；不能仅保留相同prompt文字。并发首次请求可能都尚未完成prefix，不保证计算一次。取消、异常和slot复用也要正确释放引用。

## 一个能核对的例子
3个请求分别是相同2000token prefix+50token suffix。无共享需处理6150个prefill tokens，理想缓存后是2050+50+50=2150。此处2.86是prefill token工作量比，不是实测速度比；prefill复杂度、decode、queue、cache维护和并发改变墙钟收益。

batch是否更快、命中率是否高，需实际记录cache命中tokens/查询tokens及其统计边界。把walltime变化当命中率，或用“system prompt始终缓存”的注释替代证据，都不成立。

## 配置与对照
核对目标版本是否默认启用及`--disable-radix-cache`；static memory预算包含其他项，不能直接当cache容量。启动日志中的实际分配、max_total_tokens、blocksize与eviction更有意义。
相同输入计划分别测冷/热/挤出后状态，保持采样与输出长度一致。请求端TTFT/端到端和engine缓存计数分别报告，删去无出处5×/10×、固定70%/80%门槛。
程序前端批量运行用相应program.run_batch接口，不存在通用顶层`sgl.run_batch`保证共享一次计算。
