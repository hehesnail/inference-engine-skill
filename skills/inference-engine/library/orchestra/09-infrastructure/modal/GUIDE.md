# Modal：推理容器与服务生命周期

使用当前项目已有环境配置，按任务要求选择远程函数、类或web endpoint；不自动申请GPU、开启长期容器或发布外网服务。

## 版本与资源
Modal1.0迁移将allow_concurrent_inputs替换为相应`@modal.concurrent(max_inputs=...)`，concurrency_limit改为max_containers，container_idle_timeout改为scaledown_window。具体目标版本查官方迁移文档，不能混用旧新字段。

`.local()` 在调用端本地执行，不会自动获得远程GPU。冷启动包含容器、挂载、模型下载/加载、JIT与warmup，container up不等于model ready。包名openai-whisper与其他同名whisper包要区别。

## 推理服务
固定镜像/依赖、模型revision、资源规格、volume/secret边界。并发请求数量与实际engine batch分开，扩容/缩容不破坏请求状态。读取secret按平台机制，日志不输出凭据。

记录首次请求、稳态、排队、扩容与失败。可迁移的能力是环境复现、持久模型缓存、readiness与计时拆分；不把历史GPU价格或固定冷启动数字写进验收。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `09-infrastructure/modal/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [advanced-usage](references/advanced-usage.md)
- [troubleshooting](references/troubleshooting.md)

完整历史正文及本次具体勘误保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于追溯；运行规范使用本页与专题。
