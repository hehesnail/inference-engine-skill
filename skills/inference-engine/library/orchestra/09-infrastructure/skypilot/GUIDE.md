# SkyPilot：资源配置、服务与恢复

可复用的是声明环境、资源约束、setup/run、服务readiness与任务恢复。实际字段以所装SkyPilot版本的YAML schema为准，不能把其他引擎配置塞进顶层字段。

## 配置与启动
区分envs、secrets和shell变量：YAML字符串不会一律自动展开任意`$VAR`，按官方支持的CLI/继承与run shell语义使用。accelerators可支持多种替代资源表达，检查目标schema，不将一种写法的存在误判为错误。

Sky Serve暴露端口需在resources.ports等对应字段声明，并与run服务端口、readiness一致。autostop与managed recovery字段随版本变化，先核对再执行。资源创建/发布遵循当前用户任务授权，不从示例推导新的权限。

## 恢复与性能
任务重启或spot recovery不等于模型/optimizer/request状态自动checkpoint；明确持久存储、checkpoint提交与恢复入口。模型cache下载完成与服务readiness分开。

跨云/实例比较固定源码、模型、workload和计时边界；价格、GPU可用性与网络拓扑按当次证据记录。不要把换GPU/改并发后的速度变化归结为某一个软件优化。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `09-infrastructure/skypilot/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [advanced-usage](references/advanced-usage.md)
- [troubleshooting](references/troubleshooting.md)

完整历史正文及本次具体勘误保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于追溯；运行规范使用本页与专题。
