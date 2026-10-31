> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[alibaba/atrex-kernel-agent@ffd2b70126ae](https://github.com/alibaba/atrex-kernel-agent/blob/ffd2b70126ae87a92ec48a0da0fa97b9c7bef595/src/aka/light/workspace/resources/skills/runtime-report/SKILL.md)，2026-10-31 核对。

# Runtime records — report-only mode

Use `python3 tools/sandbox.py` inside a controller-registered Episode. Direction and research
Experiment recording are disabled; do not use `update-direction` or `record-experiment`.
This does not disable automatic measurement persistence, duplicate detection, correctness checks,
or independent ABBA promotion. Framework Baseline has its own smoke-and-exit procedure.

Inspect saved measurement evidence and optional canonical `memory/v*.json` summaries when present.
Implement and measure a candidate, then leave its exact source in place (`kernel.py` or the
contracted source tree from `WORKSPACE.md`) and cite real Gateway Record IDs in `episode-report`.
No synthetic Direction, Experiment, separate plan/profile document, or Git operation is required.

- [Episode report and examples](references/journal.md): direct evidence selection and repair.
- [Saved results and source](../runtime-records/references/records.md): measurement readers, per-Shape historical best, and errors.
- Use `skills/gpu-measurement/SKILL.md` for new measurements.

Read-only Direction/Experiment history queries remain available for earlier recorded Episodes;
they do not create records or change the current policy. Never invent IDs or measured results.
Reuse saved results rather than repeating identical tasks. Accepted reports can be replayed to
repair publication, but different terminal reports are rejected. Report acceptance is not promotion.
