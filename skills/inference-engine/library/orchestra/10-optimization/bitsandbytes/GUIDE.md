# bitsandbytes：加载、低比特线性层与 offload

区分 LLM.int8、4-bit FP4/NF4、double quant、compute dtype 和训练用 optimizer。double quant 主要压缩量化元数据，不是额外准确率提升机制。

## 容量与放置
70B 参数4bit原始数据约35GB、8bit约70GB（十进制），尚无 scales、未量化层、KV/state/workspace。不能保证70B4bit整模型驻留24GB或8bit驻留40GB。CPU/disk offload 能改变 GPU 驻留，但也改变系统内存、传输与计时边界。

`offload_folder` 指定目录不等于建立正确 device_map；确认模型实际分层位置、低比特模块和 compute dtype。示例若用 torch dtype 必须先 import torch。CPU offload 与量化 backend 的支持由所用版本决定。

## 服务使用
HF 能加载不代表 vLLM/SGLang 同版本内置该格式。所核对 vLLM 快照7b77ed5 registry 没有内置 bitsandbytes；其他旧版本或插件可能支持，按运行版本核对，不将一个时间点永久推广。

比较实际质量、峰值内存、CPU/GPU传输、prefill/decode。训练 PEFT/QLoRA 的可训练参数与 optimizer 配置不自动适用于推理内存估算。细节见 [量化契约](../../../../references/quantization-engine-contracts.md)。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `10-optimization/bitsandbytes/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [memory-optimization](references/memory-optimization.md)
- [qlora-training](references/qlora-training.md)
- [quantization-formats](references/quantization-formats.md)

完整历史正文及本次具体勘误保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于追溯；运行规范使用本页与专题。
