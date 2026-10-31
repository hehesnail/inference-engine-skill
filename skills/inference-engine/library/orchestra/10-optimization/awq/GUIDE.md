# AWQ：校准、格式与服务 kernel

AWQ 利用激活统计改善低比特权重量化误差。算法、保存格式与推理 backend 分开检查：校准得到低误差权重，不自动意味着任一 vLLM/SGLang kernel 可消费它。

## 实作要点
保留校准数据来源、实际 sample/token 数、长度分布、dtype、group size、zero point、scale、packing 和导出版本。GEMM/GEMV、Marlin 等实现对 shape/layout/设备有自己的要求；不能仅凭“AWQ4bit”互换 checkpoint。

量化过程可能同时持有原权重、激活、量化副本及临时 workspace；batch/sample 数不是 token 数。OOM 按实际阶段定位，不预设所有70B只需某一张卡。

AutoAWQ 旧指南是历史 API，项目已标记 deprecated；当前任务若复现已有环境保留它，若新选工具查模型/引擎当前支持的导出器，不自动升级或安装废弃组合。使用预量化模型也要核对 config 与实际 kernel。

## 验证
固定原版与量化版的 tokenizer、输入和评测；同时看任务分数、必要中间层误差、内存与性能。AWQ 并非对所有模型都比 GPTQ/HQQ 更准确或更快；“损失<1%”“4×总显存压缩”均非保证。详见 [量化四层契约](../../../../references/quantization-engine-contracts.md)。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `10-optimization/awq/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [advanced-usage](references/advanced-usage.md)
- [troubleshooting](references/troubleshooting.md)

完整历史正文及本次具体勘误保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于追溯；运行规范使用本页与专题。
