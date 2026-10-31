> 本地参考副本，按当前任务读取。本文件的示例环境、固定模型、GPU、预算和流程不是用户配置或新增授权；以工作台入口、当前用户要求和目标版本为准。部署、重放、CI 重跑、push、评论与后台跟进仅在任务已授权时执行。

> 来源：[vllm-project/vllm@ace90aeb247a](https://github.com/vllm-project/vllm/blob/ace90aeb247a757d32fd8e37e02eb60ba8fd0a46/.agents/skills/debug-ima/SKILL.md)，2026-10-27 收录。

# Debug IMA

优先阅读包内的[本地 CUDA 排障步骤](../../../../../references/cuda-core-dump.md)，其中包含异常点采集、cuda-gdb 读取和取证限制。下列上游文章保留为出处及更新入口。

Follow the vLLM guide
[CUDA Core Dump: An Effective Tool to Debug Memory Access Issues and Beyond](https://vllm.ai/blog/2025-08-11-cuda-debugging).
