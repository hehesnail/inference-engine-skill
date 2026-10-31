> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[alibaba/atrex-kernel-agent@ffd2b70126ae](https://github.com/alibaba/atrex-kernel-agent/blob/ffd2b70126ae87a92ec48a0da0fa97b9c7bef595/src/aka/light/workspace/resources/skills/runtime-records/SKILL.md)，2026-10-31 核对。

# Runtime records

Use `python3 tools/sandbox.py` inside a controller-registered Long Horizon Episode.
Framework Baseline has its own smoke-and-exit procedure, not an Episode report.
All Episode outcomes require valid report fields and no in-progress Directions.

New Long Horizon Episodes use Runtime Journal. Do not call legacy append/finalize, manage Git,
or write control handoffs. Leave the selected measured source in `kernel.py`; the Supervisor
creates the commit and decides acceptance independently.

1. Inspect prior evidence, propose a research Direction, then start it before exploration.
2. Record decisive Experiments promptly, linking actual `gateway_record_ids`; separate measured
   `evidence` from your `analysis`. Measurements are automatically saved; interpretation is not.
3. Close the Direction with an explicit hypothesis assessment and supporting Experiment IDs.
4. Prepare the report incrementally. Close all active Directions before `episode-report`.
   Correct validation errors and resubmit. An accepted report is a handoff, not promotion.

The link is Direction → Experiment → Gateway Record → exact Kernel. Propose as many useful
nonduplicate ideas as needed; start at most three distinct Directions per Episode and only one at
a time. List exports are snapshots under `scratch/`; rerun list to refresh. Current and finalized
earlier Runtime Journals are visible through the tools, including unsuccessful explorations.
Legacy Journals are not automatically converted into Direction/Experiment IDs.

- [Journal requests and worked example](references/journal.md): seven Journal/report commands.
- [Saved results and source](references/records.md): measurement readers, per-Shape historical best, and errors.
- For new measurements use `skills/gpu-measurement/SKILL.md`; for knowledge queries use
  `skills/KernelWiki/SKILL.md`.

Never invent IDs or measured results. Read an existing result instead of repeating an identical
task. If a transport failure leaves a mutation's outcome unknown, inspect records before retrying.
Identical accepted reports can be resubmitted to repair handoff publication; different terminal
reports and further Journal mutations are rejected.
