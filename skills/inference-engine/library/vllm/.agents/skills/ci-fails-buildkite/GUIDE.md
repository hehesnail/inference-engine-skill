> 本地参考副本，按当前任务读取。本文件的示例环境、固定模型、GPU、预算和流程不是用户配置或新增授权；以工作台入口、当前用户要求和目标版本为准。部署、重放、CI 重跑、push、评论与后台跟进仅在任务已授权时执行。

> 来源：[vllm-project/vllm@ace90aeb247a](https://github.com/vllm-project/vllm/blob/ace90aeb247a757d32fd8e37e02eb60ba8fd0a46/.agents/skills/ci-fails-buildkite/SKILL.md)，2026-10-27 收录。

# Diagnosing vLLM Buildkite CI Failures

Buildkite logs are public; no login needed.

`.buildkite/scripts/ci-fetch-log.sh` saves each log as `ci-<build>-<job-name>.log`, stripped of timestamps and ANSI codes. Existing files are kept; set `CI_FETCH_LOG_FORCE=1` to refetch.

## Fetching logs

```bash
# All failed jobs in a PR's latest build (current branch's PR if omitted):
.buildkite/scripts/ci-fetch-log.sh --pr <PR>

# All failed jobs in a build (--soft also includes soft-failed jobs;
# --all fetches every finished job):
.buildkite/scripts/ci-fetch-log.sh "https://buildkite.com/vllm/ci/builds/<N>"

# One job — `gh pr checks` URLs (#<job_uuid>) and web UI URLs (?sid=) both
# work; pass "-" as a second argument to stream to stdout:
.buildkite/scripts/ci-fetch-log.sh "https://buildkite.com/vllm/ci/builds/<N>#<job_uuid>"
```

To clean an already-downloaded log with `.buildkite/scripts/ci-clean-log.sh`:

```bash
./ci-clean-log.sh ci.log
```

## Reference

See [docs/contributing/ci/failures.md](../../../docs/contributing/ci/failures.md) for the full guide: filing CI failure issues, investigating/bisecting, reproducing flaky tests, and daily triage.
