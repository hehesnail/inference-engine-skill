> 本地参考副本，按当前任务读取。本文件的示例环境、固定模型、GPU、预算和流程不是用户配置或新增授权；以工作台入口、当前用户要求和目标版本为准。部署、重放、CI 重跑、push、评论与后台跟进仅在任务已授权时执行。

> 来源：[sgl-project/sglang@e4d4a9a7de54](https://github.com/sgl-project/sglang/blob/e4d4a9a7de5454cd8b854b21af723c2155a55a3d/.agents/skills/scripted-runtime-notes/SKILL.md)，2026-10-27 收录。

# Scripted Runtime — Notes

Notes for anything related to the SGLang scripted runtime.

## When to Add an API

Tests read `r.req.*` and `t._scheduler.*` directly — there is no encapsulation boundary. A thin wrapper buys zero isolation; it only grows the surface.

Add an API only if it does real work:

1. **Control primitive** — drives the engine through a real path (`start_req`, `pause_generation`, `abort`, `evict_radix`, `exhaust_kv`). Reuse the real path; never hand-mutate state.
2. **Hook-backed** — value cannot be read from a snapshot; accumulate via `scheduler_hook.on_run_batch` or the recv proxy (`chunks_done`). Read-only; never monkey-patch; never add `*_count` to `srt/`.
3. **Multi-structure derivation, widely reused** — scans `chunked_req` + `waiting_queue` + `running_batch` + `last_batch` (`is_idle`, `status`, `batch_composition`).

Else: don't. Read `r.req.X` / `t._scheduler.X` in the test; inline single-use accessors.

Never:

- Weaken an assertion to fit a missing probe.
- Probe implementation details ("field non-None", "which branch ran") — assert the consequence.

## Other Tips

- **Engine-self-driven behavior: drive the real loop, don't call the private.** Never synchronously call a scheduler private (e.g. `scheduler._abort_on_waiting_timeout()`) from the harness/test — it runs at the wrong loop phase, bypasses the ordered `recv_requests` → `process_input_requests` injection, and can fire in states the real loop never reaches (e.g. while paused). For sweeps the engine runs itself (timeout/idle), enable the config/env and advance the loop with `yield`.
