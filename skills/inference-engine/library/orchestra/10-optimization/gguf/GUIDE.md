# GGUF：转换、量化与跨 runtime 验证

GGUF 全称 GGML Universal File，是模型与元数据容器；Q4_K、Q5_K、IQ 等是具体 tensor 编码。模型架构、tokenizer、RoPE与特殊 token 信息也必须被 runtime 正确解释。

## 转换与装载
使用支持目标架构的固定转换器，将相同 checkpoint/tokenizer revision 导出。先检查转换日志和元数据，再执行目标 quantizer；imatrix 对适用方法提供校准信息，并不是一律必选。保留源权重与转换参数，使输出可追溯。

现代 llama.cpp CMake 构建与 build/bin 命令见 [llama.cpp 指南](../../12-inference-serving/llama-cpp/GUIDE.md)。旧 make flags、未在目标CLI出现的 `--chunk` 等不作为可执行参数。其他引擎对 GGUF 的支持模型、loader与kernel不等同 llama.cpp。

## 测量与质量
4bit文件不代表总内存只有FP16的四分之一；metadata、混合精度层、KV、workspace与CPU/GPU双副本都可能存在。较小bit数不保证更快，CPU ISA、offload层数、形状与解量化路径决定性能。

用同一输入IDs/模板比较必要输出和任务分数，分开报告 prefill/decode与端到端。将格式、算法、kernel、服务接线四层对齐，见 [量化契约](../../../../references/quantization-engine-contracts.md)。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `10-optimization/gguf/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [advanced-usage](references/advanced-usage.md)
- [troubleshooting](references/troubleshooting.md)

完整历史正文及本次具体勘误保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于追溯；运行规范使用本页与专题。
