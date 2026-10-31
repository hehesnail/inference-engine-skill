# TensorRT-LLM Host / Graph 技术资料

来自 NVIDIA/TensorRT-LLM @ f405e10b730ea953961ecf5719242c581527587a，保留 Apache-2.0 版权与[许可证](../LICENSE)。这是专题资料副本，按需选一个文件；不要求整套加载。

用于 vLLM/SGLang 时先按本包 [Host/GPU 分析](../../../references/host-device-analysis.md) 和 [Graph/同步](../../../references/cuda-graphs-and-sync.md) 的口径解释。上游脚本依赖 TRT 的 NVTX 文本，部分查询不按 device/process 隔离、只筛 start 而不裁剪跨窗口事件，且以 100 µs 邻近合并 rank；直接用于其他引擎可能误判。它们保留作实现参考，未注册为本包稳定工具。

| 需要的细节 | 本地资料 |
|---|---|
| kernel busy、host prep 暴露、launch 与通信指标 | [metrics](analysis/references/metrics.md) |
| SQLite 表和字符串 ID | [schema](analysis/references/nsys-schema.md) |
| forward/phase 定位 | [iteration isolation](analysis/references/iteration-isolation-techniques.md)、[phase](analysis/references/phase-classification.md) |
| kernel 级别归因 | [kernel-level analysis](analysis/references/kernel-level-analysis.md) |
| Python、loop、alloc、graph、system 优化模式 | [optimization patterns](optimization/references/optimization-patterns.md) |
| 不同 host 热点对应的方向 | [hotspot classification](optimization/references/hotspot-classification.md) |
| 动态地址/标量/shape 的 capture 处理 | [dynamic patterns](graphs/patterns-dynamic.md) |
| graph 兼容和故障 | [compatibility](graphs/patterns-compatibility.md)、[troubleshooting](graphs/troubleshooting.md) |
| 常见同步来源及代码例子 | [sync patterns](sync-patterns.md) |

保留原表和例子是为了查技术细节，不把其中门槛、固定型号值或强制流程转成规则。特别是 host prep 与 idle 的交集不证明因果；移除占墙钟 15% 的开销，理想吞吐倍率为 1/(1-0.15)，不是已经测得 15% 收益；`non_blocking` 不能省略消费依赖；collective 的 host 阻塞要看实际 backend/版本。上游训练示例、旧 PyTorch 行为和硬编码 shape 只适用于相应条件。
