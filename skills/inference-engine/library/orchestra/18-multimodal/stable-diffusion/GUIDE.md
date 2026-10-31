# Diffusion：采样轨迹、内存与adapter

比较推理优化时固定prompt/negative prompt、seed或初始latents、resolution、batch、scheduler、step数、guidance、VAE、模型与adapter。改变scheduler/step数可能改变质量，不是等价kernel优化。

## 内存与backend
CPU/model/sequential offload按pipeline支持的放置顺序启用；先将完整模型.to(cuda)再offload可能制造加载峰值。SDPA、xFormers和attention slicing是不同路径，叠加可能减速，记录实际选中backend。

VAE slicing主要沿batch，tiling主要沿空间；检查接缝和边界质量。BF16指数范围大但尾数精度较低，不能泛称更精确。黑图先检查NaN/Inf、VAE dtype与数值溢出，不能将关闭safety checker当数值修复。

## Adapter与图
fuse LoRA把增量合入权重，unload adapter不必然撤回；用对应unfuse或恢复基线，检查重复融合、切换与并发状态。图捕获要覆盖真实shape、step与变化输入，不能用旧latents replay得出正确性结论。

详细状态和计时见 [多模态专题](../../../../references/multimodal-inference.md)。同seed不保证跨backend bitwise一致，沿任务约定比较中间tensor或最终质量。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `18-multimodal/stable-diffusion/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [advanced-usage](references/advanced-usage.md)
- [troubleshooting](references/troubleshooting.md)

完整历史正文及本次具体勘误保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于追溯；运行规范使用本页与专题。
