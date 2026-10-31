> 本地核对：基线tag不是自动已知的真实fork point；核对commit关系。基线失败先记录归因并继续可独立推进的范围；不默认squash或换wheel。四点U0/F0/U1/F1与实际Python/扩展加载身份见本地fork维护。

> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[cohere-ai/vllm-skills@2a9ed1ff5dd8](https://github.com/cohere-ai/vllm-skills/blob/2a9ed1ff5dd822a521a453f49eff84e8dd5a7071/skills/rebase-assistant/SKILL.md)，2026-10-31 核对。

# Rebase Assistant

Rebase this repository's custom fork commits onto a newer upstream vllm tag
in a safe, repeatable way.

**Notation** used throughout this document:
- **v1** — the upstream tag the branch is currently based on
- **v2** — the target upstream tag to rebase onto
- **b1** — the branch before rebase (custom commits on top of v1)
- **b2** — the branch after rebase (custom commits replayed onto v2)

**Wheel identity**: preserve the task-selected Python source and extension build. A stable wheel is usable only if its C++/CUDA interfaces match the fork changes; source changes to extensions require a matching build. Record both identities on b1 and b2. Do not switch wheels merely because a rebase is being performed.

## Step 1: Collect Inputs

Gather the following from the user before proceeding.

### Auto-detected or user-provided

1. **Current base tag (v1)**: Either provided directly by the user, or
   auto-detected via the `detect-upstream-base` skill. See
   [detect-upstream-base](../detect-upstream-base/GUIDE.md).
   If the user supplies v1 explicitly, skip detection and validate:
   ```bash
   git rev-parse "$V1" 2>/dev/null \
     || git rev-parse "$V1" 2>/dev/null \
     || echo "ERROR: tag $V1 not found"
   ```

### Required

2. **Target upstream tag (v2)**: The upstream tag to rebase onto (e.g.
   `v0.20.0`). Validate it exists:
   ```bash
   git rev-parse "$V2" 2>/dev/null \
     || git rev-parse "$V2" 2>/dev/null \
     || echo "ERROR: tag $V2 not found"
   ```

3. **Verification checks**: One or more commands to run after rebase to
   confirm correctness. Examples:
   - Test command: `.venv/bin/python -m pytest tests/path/to/test.py -v`
   - Lint command: `pre-commit run --all-files`
   - Metric threshold: "WER < 15% on transcription tests"

   Store these verbatim — they are run as-is during verification.

### Optional

4. **Reference commit**: A known-good commit where the checks pass. Used to
   diff against when debugging failures after rebase.
   ```bash
   git rev-parse --verify <reference-commit> 2>/dev/null
   ```
   If not provided, `HEAD` on **b1** is used as the implicit reference.
   Either way, Step 3 verifies the checks pass on the reference state
   before rebasing.

## Step 2: Analyze b1

Understand the custom commits on **b1** that sit on top of v1. Use the
confirmed v1 from Step 1 — the merge-base between HEAD and `$V1`
is the fork point (`FORK_POINT`), used throughout the remaining steps.

```bash
FORK_POINT=$(git merge-base HEAD "$V1")
git log --oneline "$FORK_POINT"..HEAD
COMMIT_COUNT=$(git rev-list --count "$FORK_POINT"..HEAD)
git diff --stat "$FORK_POINT"..HEAD
```

Present summary to user:
- Number of custom commits
- Files modified/added/deleted
- Overall diff summary (what the fork changes at a high level)

## Step 3: Verify checks pass on b1 (gate)

**Do not skip this step.** Before any rebase work, run every verification
check from Step 1.3 to confirm they pass on the reference state.

- If no explicit reference commit was provided, run checks on **b1** HEAD.
- If an explicit reference commit was provided, check it out and run there.

Use the `local-test-runner` skill for execution. See
[local-test-runner](../local-test-runner/GUIDE.md). Use the validated b1 extension build (see Wheel identity above).

If any check fails, **stop** and ask the user to:
- Fix the issue before proceeding, or
- Provide a different reference commit where the checks pass, or
- Proceed anyway and fix the failures on b2 after rebase

Only proceed to Step 4 once all checks pass (or the user explicitly opts to
skip).

## Step 4: Backup, Branch, and Prepare

Create safety nets, then prepare **b2**. **b1** is never modified.

1. Back up **b1**:
   ```bash
   B1=$(git rev-parse --abbrev-ref HEAD)
   git branch "backup-$B1-$(date +%Y%m%d-%H%M%S)"
   ```

2. Create **b2** for the rebase:
   ```bash
   git checkout -b "$B1-$V2"   # e.g. v0.19.1-asr-fix-v0.20.0
   ```

3. If there are multiple custom commits, ask whether to squash:

   Ask user: "There are $COMMIT_COUNT custom commits. Squash into one before
   rebasing? (recommended for cleaner conflict resolution)"

   If yes:
   ```bash
   git reset --soft "$FORK_POINT"
   git commit -m "squashed custom changes on top of $V1"
   ```

## Step 5: Rebase b2 onto v2

Replay the custom commits from **b1** onto v2, producing **b2**.

1. Rebase. **Do not pass literal `HEAD` as the third argument** — git treats
   it as a commit-ish rather than the current branch, and a successful rebase
   leaves you in detached state. Either omit the third argument (b2 is
   already checked out from Step 4.2) or pass the branch name explicitly:
   ```bash
   git rebase --onto "$V2" "$FORK_POINT"
   # or, equivalently:
   git rebase --onto "$V2" "$FORK_POINT" "$B1-$V2"
   ```
   If you do end up detached after a rebase, recover with
   `git checkout -B "$B1-$V2"` to re-point the branch to the rebased HEAD.

2. If conflicts arise, resolve them one at a time:
   - Show conflicted files: `git status`
   - For each conflict:
     - Read the conflict markers
     - Compare what changed upstream: `git diff "$V1".."$V2" -- <file>`
     - Preserve custom intent while integrating upstream changes
     - If the reference commit was provided, compare:
       `git show <reference-commit>:<file>` for guidance
   - Stage resolved file: `git add <file>`
   - Continue: `git rebase --continue`

3. If uncertain about a resolution, **stop and ask the user**.

### Conflict Resolution Rules

- Understand both sides before editing.
- Preserve custom branch intent — that's the whole point of the fork.
- Integrate upstream improvements that don't break custom behavior.
- If upstream deleted a file the fork doesn't modify, keep it deleted.
- Document non-obvious resolution decisions.

## Step 6: Verify b2

Run the checks collected in Step 1 on **b2** to confirm the rebase is correct.

1. Confirm branch shape:
   ```bash
   git log --oneline "$V2"..HEAD
   ```
   Expect: custom commit(s) on top of `$V2`.

2. Run verification checks using the `local-test-runner` skill. See
   [local-test-runner](../local-test-runner/GUIDE.md). Use the validated b2 extension build (see Wheel identity above).

3. If a check fails, debug and iterate:

   **a) Create a findings file** and update it as you investigate:
   ```bash
   mkdir -p .cache/investigation
   # Log each finding: what failed, hypotheses, what you tried, outcome
   ```
   This avoids re-treading ground across multiple fix attempts.

   **b) Compare v1 vs v2** to find relevant upstream changes:
   ```bash
   git diff "$V1".."$V2" -- <failing-file-or-dir>
   git log --oneline "$V1".."$V2" -- <failing-file-or-dir>
   ```
   If the diff is large, inspect individual commits from the log to narrow
   down which upstream change introduced the breakage.

   If the source files on the failing path are **unchanged** across v1..v2,
   the regression likely lives in a transitive dependency. Also diff
   `requirements/` between the tags and compare runtime signals — not just
   pass/fail — between the b1 and b2 test logs:
   ```bash
   git diff "$V1".."$V2" -- requirements/
   # Compare per-request token throughput, tokens generated per request,
   # warning/error log lines between b1 and b2 runs.
   ```
   Major-version bumps of transformers, torch, peft, or tokenizers
   frequently cause silent behaviour changes — especially for models
   loaded via `trust_remote_code=True`, where transformers-version-specific
   semantics (e.g. `split_special_tokens`, `use_fast`) can flip without
   any code change in vllm itself.

   **c) Check if v2 has related test cases** that test the same area — they
   may reveal expected behavior changes or new APIs:
   ```bash
   git diff "$V1".."$V2" -- tests/
   ```

   **d) Compare against the reference state** (Step 1.4 / Step 3):
   ```bash
   git diff <reference-commit> -- <failing-file>
   ```

   **e) Fix, amend, re-run**:
   ```bash
   git add -A && git commit --amend --no-edit
   ```
   Re-run the failing check via local-test-runner. Repeat until the check
   passes or escalate to user.

4. Once all checks pass, provide summary:
   - Custom commits replayed
   - Conflicts resolved (list files and decisions)
   - Check results (pass/fail for each, with log file paths)
   - Any follow-up items needing manual review

## Step 7: Cleanup

After user confirms **b2** is satisfactory:

1. Report backup branch name (snapshot of **b1**) for reference.
2. Note **b1** can be deleted if no longer needed.
3. If the user wants to push **b2**:
   ```bash
   git push origin HEAD
   ```

## Command Reference

```bash
# Detect v1 — see detect-upstream-base skill

# Analyze b1: fork point and custom commits
FORK_POINT=$(git merge-base HEAD "$V1")
git log --oneline "$FORK_POINT"..HEAD

# Backup b1, create b2, squash (optional)
B1=$(git rev-parse --abbrev-ref HEAD)
git branch "backup-$B1-<timestamp>"
git checkout -b "$B1-$V2"
git reset --soft "$FORK_POINT"
git commit -m "squashed custom changes on top of $V1"

# Rebase b2 onto v2 (b2 already checked out, do NOT pass HEAD)
git rebase --onto "$V2" "$FORK_POINT"

# Conflict resolution loop
git status
git diff "$V1".."$V2" -- <file>
git add <resolved-file>
git rebase --continue

# Abort and return to b1
git rebase --abort
```

## Usage Examples

```text
Rebase this branch from v0.19.1 to v0.20.0. Run transcription tests to verify.
```

```text
Use rebase-assistant to move custom patches to upstream/v0.20.0.
Checks: pytest tests/entrypoints/openai/correctness/test_transcription_api_correctness.py
Reference commit: c9a168488
```
