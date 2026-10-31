> 本地参考副本，按当前任务读取。本文件的示例环境、固定模型、GPU、预算和流程不是用户配置或新增授权；以工作台入口、当前用户要求和目标版本为准。部署、重放、CI 重跑、push、评论与后台跟进仅在任务已授权时执行。

> 来源：[sgl-project/sglang@e4d4a9a7de54](https://github.com/sgl-project/sglang/blob/e4d4a9a7de5454cd8b854b21af723c2155a55a3d/.agents/skills/sglang-bisect-ci-regression/SKILL.md)，2026-10-27 收录。

# SGLang Bisect CI Regression

Investigate a consistently failing CI test to find the root cause - whether it's a code regression from a specific PR, a hardware/runner-specific issue, or an environment change. Optionally reproduce the failure on a remote GPU server.

## Slash Command

`/sglang-bisect-ci-regression <test_name_or_ci_url> [ssh_target] [docker_container]`

## When to Use This Skill

- A CI test is failing consistently on main (scheduled runs)
- You need to find which PR introduced a regression
- You suspect a runner-specific or GPU-specific issue
- You want to reproduce a CI failure on a remote server

## Arguments

- **First argument (required)**: Test file name (e.g. `test_lora_tp.py`) or a GitHub Actions job URL
- **Second argument (optional)**: SSH target for remote reproduction (e.g. `user@host`)
- **Third argument (optional)**: Docker container name on the SSH target (e.g. `sglang_dev`)

If SSH target and docker container are not provided, the skill will only perform the CI log analysis and bisection, without remote reproduction. **Ask the user** for these if reproduction is needed and they weren't provided.

## Background: Scheduled CI Runs

SGLang uses the `pr-test.yml` workflow with **scheduled runs** (cron-triggered) to periodically test the `main` branch. These runs are the primary data source for detecting regressions:

- **Workflow**: `pr-test.yml` with `event: schedule`
- **Branch**: `main`
- **Dashboard**: https://github.com/sgl-project/sglang/actions/workflows/pr-test.yml?query=event%3Aschedule
- **Frequency**: Runs multiple times daily, each pinned to the HEAD of `main` at trigger time
- **Purpose**: Catches regressions that slip through PR-level CI (e.g., interaction bugs between merged PRs, hardware-specific issues)

Always use these scheduled runs (not PR-triggered runs) when bisecting regressions on `main`. The `--event schedule` filter in `gh run list` ensures you only see these periodic main-branch runs.

## Workflow

### Phase 1: Extract the Failure Signature

1. **Get the failing test details from CI logs.** If given a URL, fetch logs directly. If given a test name, find recent scheduled runs of `pr-test.yml` on `main` that failed:

```bash
# List recent scheduled runs targeting main (the primary source of truth for regressions)
# These are cron-triggered runs visible at:
# https://github.com/sgl-project/sglang/actions/workflows/pr-test.yml?query=event%3Aschedule
gh run list --repo sgl-project/sglang --workflow="pr-test.yml" --event schedule --branch main --limit 20 --json databaseId,conclusion,createdAt,headSha

# Find the job containing the test
gh run view {RUN_ID} --repo sgl-project/sglang --json jobs --jq '.jobs[] | select(.conclusion == "failure") | {name, conclusion, databaseId}'

# Get the failure details
gh run view {RUN_ID} --repo sgl-project/sglang --job {JOB_ID} --log 2>&1 | grep -E -B 5 -A 30 "AssertionError|FAIL|Error|{TEST_NAME}"
```

2. **Record the failure signature:**
   - Exact error message and assertion
   - Affected test method name
   - Model/config involved
   - Numeric values (e.g., tolerance diffs, scores)
   - Whether the failure is deterministic (same values across runs)

### Phase 2: Temporal Bisection

3. **Find the boundary between passing and failing runs.** Walk through the scheduled run history (from the `pr-test.yml` schedule runs on `main`) to identify:
   - Last known PASSING run (sha + date)
   - First known FAILING run (sha + date)

```bash
# For each scheduled run, check the specific partition/job status
gh run view {RUN_ID} --repo sgl-project/sglang --json jobs --jq '.jobs[] | select(.name == "{JOB_NAME}") | {conclusion, databaseId}'

# Verify a specific test passed or failed in a run
gh run view {RUN_ID} --repo sgl-project/sglang --job {JOB_ID} --log 2>&1 | grep -E "{TEST_NAME}|PASSED|FAILED|logprobs mismatch" | head -10
```

4. **List commits between the boundary:**

```bash
git log --oneline {LAST_PASS_SHA}..{FIRST_FAIL_SHA}
```

5. **Filter for relevant commits** that touch files related to the failing test (model layers, kernels, test utilities, etc.):

```bash
git log --oneline {LAST_PASS_SHA}..{FIRST_FAIL_SHA} -- {relevant_paths}
```

### Phase 3: Runner/Hardware Analysis

6. **Check if the failure is runner-specific.** Extract the runner identity from each failing and passing run:

```bash
# Get runner name and machine
gh run view {RUN_ID} --repo sgl-project/sglang --job {JOB_ID} --log 2>&1 | grep -E "Runner name|Machine name" | head -5

# Get GPU/driver info
gh run view {RUN_ID} --repo sgl-project/sglang --job {JOB_ID} --log 2>&1 | grep -i -E "NVIDIA-SMI|Driver Version|CUDA Version" | head -5

# Get package versions
gh run view {RUN_ID} --repo sgl-project/sglang --job {JOB_ID} --log 2>&1 | grep -E "sgl.kernel.*==|flashinfer.*==" | head -5
```

7. **Correlate runners with pass/fail outcomes.** Build a table:

| Run ID | Date | Runner | GPU Type | Driver | Result |
|--------|------|--------|----------|--------|--------|

Runner/GPU correlation is a symptom. Architecture-specific code regressions, drivers, binaries and runner state remain candidates. Compare fixed good/bad revisions on that platform before ruling out code.

### Phase 4: Code Analysis

8. **If a code regression is suspected** (including platform-specific failures), examine the candidate commits:
   - Read the changed files
   - Understand how the changes could affect the failing test
   - Look for prefill-vs-decode differences, TP-specific paths, kernel changes

9. **If a hardware issue is suspected**, analyze:
   - Kernel compatibility (CUDA compute capability)
   - Driver version differences
   - All-reduce / NCCL behavior differences
   - CUDA graph capture differences across GPU architectures

### Phase 5: Remote Reproduction (Optional)

Only if SSH target and docker container were provided.

10. **Verify the remote environment:**

```bash
ssh {SSH_TARGET} "docker exec {CONTAINER} nvidia-smi --query-gpu=name,driver_version --format=csv"
ssh {SSH_TARGET} "docker exec {CONTAINER} pip show sgl-kernel sglang flashinfer-python 2>&1 | grep -E 'Name:|Version:'"
```

11. **Reproduce the recorded revision and environment.** Use an isolated checkout/container at the failing CI SHA and image digest or dependency lock; record imported source and binary versions. Pair it with the passing SHA under the same workload/environment. Updating main or dependencies changes the experiment. Prepare missing dependencies only in an authorized isolated environment and preserve shared installations. An unavailable original image is a reproduction limit.

12. **Create a minimal reproduction script** that:
    - Uses `if __name__ == '__main__'` with `mp.set_start_method("spawn")`
    - Runs the specific failing test configuration
    - Prints key metrics (diffs, scores, outputs)
    - Exits with code 1 on failure

13. **Copy and run the reproduction script:**

```bash
scp /tmp/repro_script.py {SSH_TARGET}:/tmp/
ssh {SSH_TARGET} "docker cp /tmp/repro_script.py {CONTAINER}:/tmp/"
ssh {SSH_TARGET} "docker exec -e CUDA_VISIBLE_DEVICES=<authorized_devices> {CONTAINER} python3 /tmp/repro_script.py"
```

14. **Run control experiments** to isolate the variable:
    - If suspecting TP issue: run with TP=1 as control
    - If suspecting GPU issue: compare same code on different GPU
    - If suspecting a specific commit: test before/after that commit

### Phase 6: Report

15. **Produce a structured report:**

```markdown
## CI Regression Bisection Report

### Failure Signature
- **Test**: {test_file}::{test_method}
- **Error**: {exact error message}
- **Key metrics**: {numeric values}
- **Deterministic**: Yes/No

### Root Cause Classification
One of:
- **Code Regression**: PR #{number} introduced the bug
- **Hardware-Specific**: Fails on {GPU_TYPE}, passes on others
- **Environment Change**: New runner/driver/package version
- **Pre-existing Flakiness**: Intermittent, not a new regression

### Evidence
| Condition | Result |
|-----------|--------|
| {condition1} | PASS/FAIL |
| {condition2} | PASS/FAIL |

### Timeline
- {date}: Last known pass ({sha}, {runner})
- {date}: First known fail ({sha}, {runner})
- {date}: Confirmed reproduction on {server}

### Recommended Fix
- **Short-term**: {workaround}
- **Long-term**: {proper fix}
```

## Key Patterns to Recognize

| Pattern | Hypothesis to test (not a diagnosis) |
|---------|-----------|
| Same SHA passes on runner A, fails on runner B | Hardware/runner-specific |
| All runners fail after commit X | Candidate code/environment change; verify paired revisions and actual test execution |
| Intermittent - same runner sometimes passes/fails | Flaky test or race condition |
| Prefill OK but decode fails | Decode state/cache/graph/sampling/collective candidate; locate first divergence |
| Works with TP=1, fails with TP>1 | Parallel-path candidate; match memory, kernels and batch conditions |
| Exact same numeric diff every time | Reproducible difference; check precision, tolerance and reference before classifying a bug |

## Important Notes

- **Always check runner identity** before concluding it's a code regression. Many "consistent" failures are actually runner-specific.
- **Test partition assignments change over time** as tests are added/removed. A test may move between partitions, landing on different runner types.
- **H200 runners** use `/root/actions-runner/` path and machine names like `gpu-h200-worker-*`. Non-H200 runners use `/public_sglang_ci/runner-*` paths.
- When running remote reproduction, use `run_in_background` for long-running tests and check output with `TaskOutput`.
- Container environments may be stale - always verify package versions match CI before drawing conclusions.
