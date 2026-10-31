# HQQ：量化元数据与后端转换

HQQ 的量化算法与优化执行 backend 分开。模型能经 Python dequantize 得到权重，不等于服务引擎支持直接运行该 checkpoint。

## 源码契约
核对过的 `HQQLinear.set_backend` 使用 backend enum 的值，不能把任意字符串当成通用参数。核心 backend 与转换到特定 fused/packed kernel 是不同步骤。scale/zero、shape、group、axis 等在 metadata 中管理，不应假设统一存在于顶层 `layer.scale`/`layer.zero`。

axis 与 group reshape 定义共同决定分组；所核对实现中 axis=1 与 axis=0 使用不同 reshape，不能用“沿列/沿行”口头说明替代实际 tensor 检查。保存、恢复、量化元数据自身的压缩与设备迁移需要一起验证。

## 引擎接线
固定 vLLM 快照7b77ed5的 quantization registry 没有内置 hqq；不要复制 `--quantization hqq` 或顶层 `LLM(..., quantization_config=...)` 后宣称官方支持。插件/第三方适配若注册了实现，需单独记录其身份和实际 dispatch。

性能比较固定输入、warmup、batch、backend与同步边界；一次计时包含首次编译而另一次不包含会失真。看 [量化契约](../../../../references/quantization-engine-contracts.md) 与 [kernel 接线](../../../../references/kernel-engine-integration.md)。配套参考中的旧接口先对照目标 HQQ 源码。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `10-optimization/hqq/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [advanced-usage](references/advanced-usage.md)
- [troubleshooting](references/troubleshooting.md)

完整历史正文及本次具体勘误保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于追溯；运行规范使用本页与专题。
