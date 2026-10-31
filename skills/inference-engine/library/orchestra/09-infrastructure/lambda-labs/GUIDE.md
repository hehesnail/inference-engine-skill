# Lambda 云资源与分布式推理

云实例规格、价格、配额与API是时变信息，使用时查官方当前内容。已有服务器能完成任务时沿用当前配置，不因指南示例自动创建实例。

## 环境与进程
固定镜像、驱动/CUDA/PyTorch与引擎版本，验证拓扑、挂载和模型身份。SLURM的tasks-per-node与torchrun的nproc-per-node需要配合：8个SLURM task各启动8进程会变成64进程/节点，而不是预期8卡训练/推理。world/rank对应实际进程而非只数节点。

API client上下文管理器退出后不再复用已关闭的session。认证与资源状态分别排查，避免把一次网络失败当作权限不足。

## 实验与清理
下载、CPU准备与GPU实验分开；按已授权的资源和时段执行。故障优先沿进程、依赖、网络和日志定位，不把reboot整个实例作为默认修复。保留源代码/镜像和结果，清理范围只含本任务拥有的临时资源。

性能与成本按真实GPU时长、实际完成tokens和失败率计算，不沿用示例价格或任意固定速度。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `09-infrastructure/lambda-labs/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [advanced-usage](references/advanced-usage.md)
- [troubleshooting](references/troubleshooting.md)

完整历史正文及本次具体勘误保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于追溯；运行规范使用本页与专题。
