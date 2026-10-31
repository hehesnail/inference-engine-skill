# alibaba/atrex-kernel-agent

固定提交：[ffd2b70126ae87a92ec48a0da0fa97b9c7bef595](https://github.com/alibaba/atrex-kernel-agent/tree/ffd2b70126ae87a92ec48a0da0fa97b9c7bef595)。

先读本地适配：[不可变测量和报告](../agentic-kernel-workflows.md)。

当前采用 19/19 项。采用项正文逐篇语义审阅；相关源码和附属资源的核对范围单列，不能推广成整仓每行已验证。技术修订与限制见 [语义核对](../semantic-audit.md)。

| 专项 | 用途 |
|---|---|
| [opt-trace-mining](../../library/atrex/gpu-wiki/skills/opt-trace-mining/GUIDE.md) | 代码事件/测量/捕获各自权威来源；ratchet与单次失败事实区分；候选结果可追commit。 |
| [query_bridge_agent](../../library/atrex/gpu-wiki/skills/query_bridge_agent/GUIDE.md) | query intent不混入检索结果；未知硬件用null；测量症状需profile证据。 |
| [session-trace-mining](../../library/atrex/gpu-wiki/skills/session-trace-mining/GUIDE.md) | T1原始stdout与T2/T3自写记录分开，排除agent prose与目标数字作证据；跨session pairing。 |
| [wiki-gate](../../library/atrex/gpu-wiki/skills/wiki-gate/GUIDE.md) | 先match后人/agent语义判定，再insert/confirm/conflict，重复确认幂等。 |
| [micro-analysis](../../library/atrex/plugins/micro-analysis/skills/micro-analysis/GUIDE.md) | 最小能区分假说的probe；实际指令、数据layout、依赖链、资源变化与探针扰动；同操作最终性能单独验证。 |
| [aka-execution](../../library/atrex/src/aka/legacy/execution/skills/aka-execution/GUIDE.md) | 入口绑定、argv边界、输入allowlist、返回码与局部输出保留、未知提交复用job身份。 |
| [autonomous-gpu-kernel-timeline](../../library/atrex/src/aka/legacy/execution/skills/autonomous-gpu-kernel-timeline/GUIDE.md) | 探针快照不晋升；owner唯一、overflow拒绝、单launch、原始计时重算；probe-free终验。 |
| [ppu-acu-joint-profile](../../library/atrex/src/aka/legacy/execution/skills/ppu-acu-joint-profile/GUIDE.md) | ACU device-global与timeline owner-local不同；无共同clock不能假定对齐；hash与身份匹配、interval union避免双计。 |
| [gen-plan](../../library/atrex/src/aka/legacy/workflow/skills/gen-plan/GUIDE.md) | 固定候选与证据再独立review；原draft完整保留，scope与假说链清楚。 |
| [gpu-kernel-baseline](../../library/atrex/src/aka/legacy/workflow/skills/gpu-kernel-baseline/GUIDE.md) | 语义/shape/stride/mask/写回/累积dtype先明确，再不可变evaluator测量。 |
| [gpu-kernel-episode-loop](../../library/atrex/src/aka/legacy/workflow/skills/gpu-kernel-episode-loop/GUIDE.md) | 从incumbent和失败记录选假说；保留测量hash、恢复clean候选；Supervisor独立ABBA才晋升。 |
| [aka-execution](../../library/atrex/src/aka/light/execution/skills/aka-execution/GUIDE.md) | 可靠执行checkpoint记录远端job；unknown不盲重投；Eval超时≠infra，不能换Dev绕过。 |
| [KernelWiki](../../library/atrex/src/aka/light/workspace/resources/skills/KernelWiki/GUIDE.md) | 真实product/runtime arch分开，缺失不拿scheduler alias替代；材料使用与只检索区分。 |
| [atrex-gpu-gateway](../../library/atrex/src/aka/light/workspace/resources/skills/atrex-gpu-gateway/GUIDE.md) | eval/check/profile能力边界、native CUDA manifest ABI、依赖commit和制品hash、current stream、不隐式copy。 |
| [autonomous-gpu-kernel-timeline](../../library/atrex/src/aka/light/workspace/resources/skills/autonomous-gpu-kernel-timeline/GUIDE.md) | light timeline明确私有evaluator receipt不可见时只diagnostic；候选由Supervisor提交。 |
| [gpu-measurement](../../library/atrex/src/aka/light/workspace/resources/skills/gpu-measurement/GUIDE.md) | 保存Kernel/Gateway Record精确身份，源拒绝与infra区分，memory SOL非DRAM；测量与解释分开。 |
| [ppu-acu-joint-profile](../../library/atrex/src/aka/light/workspace/resources/skills/ppu-acu-joint-profile/GUIDE.md) | 同legacy PPU方法，新增scratch完整证据树与相对路径以跨remote同步。 |
| [runtime-records](../../library/atrex/src/aka/light/workspace/resources/skills/runtime-records/GUIDE.md) | Direction→Experiment→Gateway Record→Kernel证据链，未结方向先close；report handoff≠promotion。 |
| [runtime-report](../../library/atrex/src/aka/light/workspace/resources/skills/runtime-report/GUIDE.md) | report-only时仍自动记录/去重/独立晋升，无须虚构Direction/Experiment或额外文档。 |
