> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[alibaba/atrex-kernel-agent@ffd2b70126ae](https://github.com/alibaba/atrex-kernel-agent/blob/ffd2b70126ae87a92ec48a0da0fa97b9c7bef595/src/aka/light/execution/skills/aka-execution/SKILL.md)，2026-10-31 核对。

For prepared host commands, use the selected `aka execution --profile reliable` entry point and its `--help` request format. The selected controller executor retains request checkpoints and known remote job IDs outside the Agent workspace. Retries apply only to confirmed admission/transport failures. An unknown submission outcome must be reconciled by the operator, not blindly resubmitted. An Eval execution timeout is a candidate failure, not an environment outage. Do not switch a failed Eval to Dev.

This selection does not replace a legacy Agent's direct `tools/sandbox.py`, or install the light HTTP ToolHost or Journal. The light Workflow separately provides its authenticated HTTP client and the `gpu-measurement` and `runtime-records` Skills. Do not assume that loading this Skill changes every tool's transport.
