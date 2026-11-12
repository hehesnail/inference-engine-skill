# 排障与代码 review

完整排障流程见[本地 SGLang 指南目录](catalog.md#sglang-专项开发与排障)和[BBuf incident 指南](../library/bbuf/skills/sglang-prod-incident-triage/GUIDE.md)。[review 经验与查询说明](../library/bbuf/skills/sglang-humanize-review/GUIDE.md)、[模型 PR 历史](../library/bbuf/model-pr-optimization-history/GUIDE.md)及其脚本已收录；大语料已安装在服务器，使用[工具入口](tools.md)即可查询，无需重新下载。

## 保存能区分假设的最小证据

锁定实际源码/镜像、模型/tokenizer、请求、并行拓扑及有效配置。记录第一个失败和此前的相关日志，不只保留最终 traceback。采样参数、stop 条件、工具/多模态输入和流式 chunk 都可能是触发条件。

| 症状 | 优先验证 |
|---|---|
| 输出或 usage 不对 | 沿请求归一化、模板/token IDs、调度结果、输出处理、JSON/SSE 逐层对照；内容和统计分别检查 |
| 缓存相关错误 | 冷/热缓存、部分命中、恢复/淘汰状态；定位 key、block/page ownership、有效长度与释放时机 |
| OOM | 失败阶段、各 rank 实际容量、KV/状态池、graph/workspace；区分加载失败与运行时峰值 |
| 分布式 hang | 各 rank 是否仍存活、首次异常、最后进展与通信序列；查首个状态分歧，而不把等待处直接当根因 |
| CUDA IMA / device assert | 报错可能延迟到同步点；用隔离复现、官方 crash/IMA 指南和具体 kernel 证据定位 |
| 延迟或吞吐退化 | 先对齐工作负载、配置、缓存、版本与计时口径，再读对应阶段 trace |

分布式问题对齐同一步的 request/batch、shape、collective 次序与参与者；PP 还要对齐 send/recv。一个 rank OOM 会使其他 rank 看起来像通信挂起，通信或驱动故障也需要证据区分。

加日志优先使用已有 CPU 元数据。把 GPU tensor 转 CPU、`.item()` 或 `.tolist()` 会引入同步，可能掩盖竞态或卡在不安全位置；记录这些观测本身造成的变化。debug 开关是定位工具，不自动成为生产修复。

停止或重启只针对本次实验拥有的服务。历史 skill 的 `pkill -f`、全机进程清理和不带目标的缓存 reset 不作为共享服务器操作模板。当前授权不足时继续做 CPU 分析和准备，仅把确实依赖新增资源的步骤留待授权。

数值问题先检查参考自身重复性和观测是否改变融合，按[数值漂移排查](numerical-drift-debugging.md)保留原 dtype、真实 launch 配置和单变量回放。输入相同的局部因果证据、整网约定回归、性能测量分别支持不同结论；不要以“浮点误差”结束归因，也不要把更高精度当作必然修复。

异步任务停在等待操作，只定位了等待方；结合生产者、依赖进程、异常和最后进展判断原因。需要恢复时，先核对已有持久记录能否可靠标识已完成工作及其配置，避免重复执行或误复用。监督状态与实际产物矛盾时保留证据并对账；统计区分各次尝试成本和整体历时。例如，CPU 判分的信号量等待与 durable journal 恢复属于这一类案例，不要求其他任务也采用同一恢复机制。

## Review 与修改

读取目标 head 的 diff、调用方与相关测试，必要时与 base 复现同一输入。不要拿其他 checkout 的符号、默认值或行号对 PR 下结论。性能变化先检查实际执行了哪个后端；数值和存储契约的失败不能仅因发生在依赖中就排除。

优先追踪：输入进入哪里、状态由谁持有/何时修改/何时释放、下游如何消费。尤其注意 parser/usage 的服务链路、KV 生命周期、rank 间状态、graph buffer 复用及 fallback。用一个具体例子说明改动前后的结果。

历史 PR 是线索。只有当前调用关系、重复优化或维护者取舍需要背景时，才查相关模块/模型的历史；优先当前官方 PR 和源码，BBuf 的历史语料可辅助检索。引用旧讨论保留其日期并复核现在的实现，不要求每次 review 全量扫描语料，也不为了模仿审阅风格强制画图。

发现问题时给出触发条件、可核对位置、影响与修复方向；区分确认的回归、尚待验证的假设和风格建议。未发现问题就直接说明，并交代与改动有关的验证缺口。已有测试通过不能代替新失败路径的复现。

实现后的结束条件是本任务相关路径已修复并验证，或明确记录真实环境限制。重试由故障假设驱动；间歇问题按观测发生率和资源预算决定重复量，不把固定“十次通过”当普适证明。

## 具体执行路径

算子原位/stride、custom op、guard/fallback 和 CUDA 正确性工具读 [kernel 接线](kernel-engine-integration.md)；异步 D2H、stream/event、graph buffer 生命周期读 [Graph/同步](cuda-graphs-and-sync.md)；TP/EP 与 PD 的能力、传输和语义分层读 [平台与传输](platforms-and-transport.md)。上游升级后的组合回归读 [fork 维护](fork-maintenance.md)。
