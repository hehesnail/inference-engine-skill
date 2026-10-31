# Host / GPU 时间线与 decode 步预算

用于 CPU 调度、输入准备、launch、同步或通信是否拖慢推理的问题。先按 [profiling-memory.md](profiling-memory.md) 确认 trace 有效，再分析选中的 rank、device、阶段与工作负载。这里吸收 TensorRT-LLM 的 exposed-host 分析和 gpu-perf-tune 的 decode 预算；修正了跨设备混算、窗口裁剪和因果推断边界。

## 选对窗口和事件

- 用请求 ID、forward/step 标记、输入输出 token 数与实际调度记录确定窗口；prefill、decode、verify、warmup、drain 分开。不要用“kernel 最密集处”代替有效 forward。
- 同一窗口内按设备计算。多个进程共享一张设备时，分别给出目标进程活动和设备总活动；其他进程填满空洞并不说明目标引擎有效利用了 GPU。
- CUDA Graph 的 node trace 可用于 kernel 统计；仅有 graph 总区间时，它包含图内等待，不能当成全程有 kernel 执行。CPU 的 `cudaGraphLaunch` 区间更不能代替 GPU 图区间。
- 有 memcpy/memset 时，同时报告 kernel 活动和包括 copy engine 的设备活动。kernel 空洞可能正发生数据搬运；NCCL kernel 正在等待也仍占据 kernel 时间。
- 缺 GPU 事件、缺 graph 内部事件、丢 CUPTI 记录和缺 NVTX 是不同的未知状态。先读采集日志和 schema；不能统一归因成某个 CUDA/driver 版本冲突。
- 单请求连续生成有助于分离 decode 步开销，但上下文长度仍须匹配被分析的问题。小 prompt 的 c=1 实验不能替代长 KV 或并发混合负载。

## 可核对的时间口径

对半开窗口 `W=[a,b)`，将每个事件裁剪为 `[max(start,a),min(end,b))`，丢弃空区间，再求并集：

| 量 | 定义与用途 |
|---|---|
| kernel duration sum | 各 kernel 裁剪后时长之和；用于工作量排名，多 stream 时可超过墙钟 |
| kernel busy | 同设备 kernel 区间并集的长度 |
| observed device busy | 同设备 kernel、memcpy、memset 区间并集；依赖采集完整性 |
| uncovered time | `len(W) - observed device busy`；先称未覆盖时间，确认完整采集后才能解释 idle |
| host-prep overlap with uncovered time | host-prep 区间并集与上述未覆盖区间的交集；用于定位候选 |
| exposed critical-path delay | 通过生产者/消费者、CUDA correlation、事件/stream 依赖证明下一段工作必须等待的延迟 |

存在 graph 总区间而缺 node 时，额外报告 graph-envelope coverage，保留图内真实 busy 未知；不要把它加成精确 busy。host-prep 的父子 NVTX 或多线程嵌套同样先求并集，避免重复相加。

例：1,000 µs 的窗口中，kernel 在 `[0,400)` 和 `[650,1000)` 执行；host prep 在 `[100,650)`。kernel busy 为 750 µs，未覆盖 250 µs，prep 总计 550 µs，重叠未覆盖区间为 250 µs。还需要证明 650 µs 处的 launch 等待这份输入，才能把 250 µs 归给它。若完全消除且其他工作不变，理想延迟减少 25%，吞吐上限倍率为 `1000/750=1.333`；这不是已测收益。

跨 rank 使用同一步语义 ID 和明确时钟关系对齐，保留最早/最晚到达与 rank skew。不要因为时间相差小于 100 µs 就合并两个 rank；也不要将所有设备事件并集当成一张 GPU 的利用率。

## Nsight Systems SQLite 查询入口

先只读检查实际表与列；不同版本未必齐全：

```python
import sqlite3
from pathlib import Path
db = Path(trace_path).resolve()
conn = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)
tables = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table'"
).fetchall()
columns = conn.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)").fetchall()
```

常见表是 `CUPTI_ACTIVITY_KIND_KERNEL`、`RUNTIME`、`MEMCPY`、`MEMSET`、`GRAPH_TRACE`、`NVTX_EVENTS` 和 `StringIds`。名称列通常需要 `shortName/nameId/textId → StringIds.id`；NVTX 也可能直接存 `text`。检查后使用对应字段，不能把整数 ID 当 kernel 名。

读取跨窗口事件用 `start < b AND end > a`，再裁剪；仅查 `start >= a` 会漏掉跨入窗口的 kernel。还要按实际 schema 中的 device、process/context 标识过滤。correlation ID 须结合进程与上下文解释，不能只按一个整数跨 rank 连接。

## 从现象到可验证的修改

| 现象 | 还要确认什么 | 候选方向 |
|---|---|---|
| 下一步 launch 前有 host 空洞 | 队列确有可运行请求；输入准备、IPC、tokenizer、scheduler 中哪个在前置依赖上 | 合并元数据转换、批量准备、复用静态对象、减少 Python 循环 |
| `.item()` / D2H 后继续 launch | CPU 是否必须立刻消费；谁拥有 buffer、何时完成 | 延后消费、批量拷贝、GPU 端处理；细节见 [同步与图](cuda-graphs-and-sync.md) |
| CPU launch 密集，GPU 很多短 kernel | 正式路径的 graph coverage、padding、piecewise 边界和 eager tail | 先查现有融合/图路径是否实际启用，再写新 kernel |
| NCCL 时间长 | 各 rank 到达时间、消息量、通信与计算重叠、实际 NIC/NVLink 路径 | 先分清 rank skew、通信本身和暴露的通信尾部 |
| 某 kernel 累计耗时高 | 是否在关键路径、shape/次数、dispatch/fallback、内存与计算信号 | 用 [NCU](nsight-compute.md) 深入代表性 launch |
| Python 函数 CPU 总耗时高但被 GPU 覆盖 | 优化它是否会改变下一次提交时间 | 不把 line_profiler 排名当成端到端收益排序 |

`time ≈ step_wall / committed_tokens_per_step` 只在同一稳态、单请求语义下可与 decode TPOT 对照。Spec decode 要用真正提交的 token 数，包含拒绝、回退和额外 target token 的实际语义；batch 总 token 数不能当每请求 token 数。客户端 TPOT 若排除首 token，也应排除相应窗口，说明差异而非强制一个固定误差门槛。

对照报告保留窗口、覆盖率、未知事件、source/dispatch 映射和一个能区分原因的实验。详细原始参考在 [TRT host 索引](../library/trtllm/host/GUIDE.md)；这些脚本按 TRT 标记编写，不直接作为 vLLM/SGLang 的通用解析器。
