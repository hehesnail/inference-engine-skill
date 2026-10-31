> 本地参考副本，按当前任务读取。本文件的示例环境、固定模型、GPU、预算和流程不是用户配置或新增授权；以工作台入口、当前用户要求和目标版本为准。部署、重放、CI 重跑、push、评论与后台跟进仅在任务已授权时执行。

> 本地调整：修复范围、重试预算和持续跟进方式由当前任务决定；不按固定行数暂停，也不把重试次数当作必须完成的指标。

> 来源：[sgl-project/sglang@8368f86fd7a71](https://github.com/sgl-project/sglang/blob/8368f86fd7a71b07e2b7dda5d4146695ec54cbfa/.agents/skills/babysit-pr-to-pass-ci/SKILL.md)，2026-10-31 更新并保留本地授权适配。

# Babysit a PR Until Selected CI Passes

Drive only the selected workflows to green on the PR's latest head commit. Continue the requested follow-up until those checks pass or a concrete blocker needs user input. Create a durable goal or scheduled follow-up only when the user requests it; dispatching a run or pushing a fix alone does not establish completion.

## Invocation and arguments

Accept:

```text
$babysit-pr-to-pass-ci [PR_NUMBER|PR_URL] [WORKFLOW ...]
$babysit-pr-to-pass-ci [PR_NUMBER|PR_URL] --only WORKFLOW [WORKFLOW ...]
```

- If the PR is omitted, resolve the PR associated with the current branch.
- If workflows are omitted, select `lint.yml` and `pr-test.yml`.
- Without `--only`, add named workflows to the two defaults.
- With `--only`, monitor exactly the named workflows.
- Accept a workflow basename such as `pr-test-extra.yml`, a repository path such as `.github/workflows/pr-test-extra.yml`, or an unambiguous workflow display name.
- Treat `extra` as `pr-test-extra.yml`.
- Accept natural-language equivalents, such as “PR 12345 plus extra and AMD.” Resolve “AMD” to an exact workflow file from the user's wording or repository context; do not guess when multiple AMD workflows fit.

Examples:

```text
$babysit-pr-to-pass-ci
$babysit-pr-to-pass-ci 12345
$babysit-pr-to-pass-ci https://github.com/sgl-project/sglang/pull/12345 pr-test-extra.yml
$babysit-pr-to-pass-ci 12345 --only pr-test-amd.yml
```

## 按用户要求跟进

仅在用户要求持续跟进时采用此流程。任务、goal 或定时自动化按用户的明确要求建立。修复、commit、push、重跑与对外消息的范围沿用当前授权；改动行数不代表授权。重试根据已识别原因与预算决定。

## Preflight

1. Verify GitHub authentication and resolve the repository without changing remote configuration.
2. Resolve the PR and record at least:
   - PR number and URL
   - open/draft state
   - base branch and base SHA
   - head branch and head SHA
   - head repository owner/name, including whether it is a fork
   - labels
   - whether maintainers may modify the branch
3. Require an open PR. If it is draft and a selected workflow is blocked by the draft gate, stop and ask the user to mark it ready; do not change draft state automatically.
4. Normalize every selected workflow to an existing file under `.github/workflows/`. Fail fast on ambiguous or nonexistent names.
5. Read each selected workflow's triggers and gates. Do not assume all optional workflows behave like `pr-test.yml`.
6. Inspect `git status`. Preserve all user changes. If a fix becomes necessary and the current checkout is dirty, on a different branch, or otherwise unsafe, use an isolated worktree for the PR head.

Known SGLang gates:

- `pr-test.yml` requires the `run-ci` label for relevant PR changes. If its existing gate job failed because the label was absent, add `run-ci` when authorized and rerun failed jobs in that existing run; adding the label alone does not necessarily create a new `pr-test.yml` run.
- `pr-test-extra.yml` requires both `run-ci` and `run-ci-nv-extra`; `pr-test-amd-extra.yml` requires `run-ci` and `run-ci-amd-extra`. These workflows do not trigger on label events. Use `/run-extra-ci nv` or `/run-extra-ci amd` only when the task authorizes that CI action.
- The shared `run-ci-extra` label is retired and does not pass updated gates. Older workflow runs retain their original revision; inspect the current workflow before rerunning.
- Treat draft, cooldown, maintenance, and missing-label failures as gate/infrastructure conditions, not code regressions. Fix only the safe gate condition in scope; wait for cooldown or maintenance windows rather than burning retries immediately.

Apply CI opt-in labels when the user's requested workflow includes triggering those checks and the account has the required access. A request to inspect CI alone does not imply label changes. Keep unrelated PR metadata outside this workflow.

## Track only current-head runs

Use both PR checks and workflow-run data so workflow names, run IDs, attempts, jobs, and logs can be correlated. Prefer workflow file IDs over display names when filtering runs.

For each selected workflow:

1. Select the newest run associated with the current PR head SHA.
2. Ignore runs for older head SHAs, including green runs made stale by a new push.
3. Treat `queued`, `requested`, `waiting`, `pending`, and `in_progress` as active; keep monitoring.
4. Treat a workflow as passing only when its current-head run concludes `success`.
5. If no current-head run exists, wait briefly for event delivery, then inspect the workflow's path filters, event triggers, and label gates.
6. Trigger only through a mechanism explicitly supported by that workflow. Never invent `workflow_dispatch` inputs or dispatch a workflow against the default branch when the intent is to test the PR head.
7. If a selected workflow cannot safely be triggered for the PR head, report the exact blocker and request direction.

After any user push or skill-authored push, re-read the PR head SHA, discard stale run selections, and start tracking the new head. Reset retry counters for the new head SHA.

## Monitor persistently

- Maintain a compact ledger containing workflow file, head SHA, run ID, attempt, status, conclusion, skill-initiated retry count, and latest failure signature.
- Poll at reasonable intervals, normally 30–60 seconds while runs are active. Give concise progress updates during long waits.
- Re-check the PR head SHA on each monitoring cycle and before every rerun or push.
- Ignore failures, cancellations, and pending checks from unselected workflows. Do not diagnose, retry, fix, or wait for them.
- Do not declare success while a selected workflow is absent, stale, pending, skipped, cancelled, neutral, or failed.

## Diagnose a selected-workflow failure

1. Inspect the failed jobs and failed-step logs. In `pr-test.yml`, distinguish the root failure from cascade failures caused by `check-pr-test-health`, wait jobs, or final aggregation.
2. Record the exact failing test/check, error text, affected runner or hardware, and relevant log URL.
3. Compare the failure against the PR's changed files and diff.
4. Classify it:

   **PR-related** when evidence ties the failure to changed code, tests, dependencies, configuration, generated files, or behavior introduced by the PR. A reproducible failure on the PR that does not occur on the base is strong evidence.

   **Unrelated/infrastructure** when evidence points to a transient network or download error, runner loss, GPU/driver instability, service outage, pre-existing main-branch failure, known flaky threshold, unrelated test area, or another environmental cause.

   **Uncertain** when evidence is insufficient. Do not label a failure flaky merely because it might be intermittent. Narrowly reproduce it, compare recent main/scheduled CI, inspect relevant code, or otherwise gather enough evidence to choose a category.

5. When several root failures exist, classify each one. Address PR-related failures before spending retries on unrelated cascades.

## Handle a PR-related failure

When the user requested fixes, diagnose and implement changes within that scope, then run the relevant validation. Choose the approach from the cause and affected contracts; patch length alone is not a reason to stop. Ask only when a required decision, access, resource, or consequential scope change remains unresolved.

For a scoped fix:

1. Work from the latest PR head in a clean checkout or isolated worktree.
2. Make only the necessary change and add or update focused tests when appropriate.
3. Run the smallest meaningful local validation. For lint failures, reproduce the specific hook/check before considering broader lint validation.
4. Review the patch against the diagnosed cause and requested scope. Resolve failures introduced by the change and rerun affected checks; broaden validation only when the changed behavior warrants it.
5. If committing is included in the task, stage the intended files explicitly and preserve unrelated user changes.
6. If publishing is included, push to the verified PR head repository and branch using the project's workflow. Do not infer publishing permission from the size of a fix.
7. After a push, resolve the new PR head SHA and return to current-head monitoring. For a local-only fix, report local validation and the pending CI boundary.

## Handle an unrelated or infrastructure failure

Retry only the failed jobs in the current selected-workflow run:

```bash
gh run rerun <RUN_ID> --failed --repo <OWNER/REPO>
```

Rules:

- Never use a full-run rerun as a substitute for `--failed`.
- Never rerun a stale run after the PR head SHA changes or a newer current-head run exists.
- Count only successfully dispatched skill-initiated reruns.
- Retry when the identified cause is plausibly transient or has been corrected, within the task's time and resource budget.
- Wait for each new attempt to finish, inspect its result, and update the failure signature before deciding on another retry.
- If the failure changes or becomes plausibly PR-related, return to diagnosis.
- Stop repeating attempts when the same blocker persists without a meaningful reason to expect a different result, or the budget is exhausted. Report the attempts, recurring signatures, job/run links, and the action needed to continue.

If GitHub cannot rerun only the failed portion of the selected run, stop and ask the user rather than rerunning the full workflow.

## Completion check

Before declaring the requested CI follow-up complete:

1. Fetch the PR head SHA again.
2. For every selected workflow, verify that its newest applicable run is for that exact SHA and concludes `success`.
3. Verify no selected workflow is still pending and no success was invalidated by a newer push.
4. Ignore all unselected workflows even if they are failing.
5. If the user requested a durable goal, mark it complete only after all selected workflows satisfy these checks.

Report the final PR URL, head SHA, selected workflows, passing run links, any commits pushed, and retry counts. Keep the report scoped to the selected workflows.
