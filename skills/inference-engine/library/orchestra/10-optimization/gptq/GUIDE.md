# GPTQ：校准输入、group 与 runtime 支持

GPTQ 使用校准激活近似权重误差传播并逐步量化。实际效果依模型、校准分布、group、desc_act、格式与 kernel 而变，不能按方法名给固定质量排名。

## 校准与导出
所核对 AutoGPTQ 实现的 quantize 接受带 `input_ids`/`attention_mask` 的样本字典，不是裸文本列表。先构造 BaseQuantizeConfig 再加载待量化模型，保留精确校准输入和导出参数。`use_marlin` 不是该版本 quantize 的通用关键字；Marlin 兼容/转换属于相应加载或导出路径。

更小 group 通常需要更多 scale/zero metadata，不是更少；scale 的 dtype 不等于4-bit权重。group_size=-1 的含义查实现（通常整行/不分组），不能解释成 per-channel 的任意同义词。desc_act 还涉及重排与 backend 支持。

## 推理与内存
checkpoint 格式、weight packing、scale/zero、g_idx、设备与实际 kernel 匹配。`device_map` 的整数列表不是任意框架都支持的模型分配表达式。不要引用不存在的“Llama2-405B”作容量实例。

算原始权重字节后再加 metadata、KV、临时和图池；70B4bit的原始权重约35GB只是下界。质量和速度使用当前 workload 配对测量，参见 [量化契约](../../../../references/quantization-engine-contracts.md)。旧 AutoGPTQ 范例不等于当前推荐安装路线。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `10-optimization/gptq/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [calibration](references/calibration.md)
- [integration](references/integration.md)
- [troubleshooting](references/troubleshooting.md)

完整历史正文及本次具体勘误保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于追溯；运行规范使用本页与专题。
