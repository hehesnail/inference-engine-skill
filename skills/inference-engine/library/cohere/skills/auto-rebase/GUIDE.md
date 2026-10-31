> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[cohere-ai/vllm-skills@2a9ed1ff5dd8](https://github.com/cohere-ai/vllm-skills/blob/2a9ed1ff5dd822a521a453f49eff84e8dd5a7071/skills/auto-rebase/SKILL.md)，2026-10-31 核对。

# Auto Rebase

Check whether a newer upstream vllm release exists and, if so, rebase the
current fork branch onto it automatically. Uses the same notation as
`rebase-assistant`: **v1**/​**v2** for upstream tags, **b1**/​**b2** for the
branch before/after rebase.

## Step 0: Prerequisites

Verify required tools are installed and authenticated before starting.

```bash
# gh CLI — needed to query upstream releases
gh auth status 2>/dev/null || echo "ERROR: gh not authenticated"
```

If `gh auth status` fails, prompt the user to run `gh auth login` and wait
for confirmation before proceeding. The git-tag fallback in Step 2 works
without `gh`, but release metadata (pre-release flags, release notes) is
only available through the GitHub API.

## Step 1: Detect Current Upstream Base

The current branch is **b1**. Use the `detect-upstream-base` skill to
identify **v1** — the upstream tag that **b1** is based on. See
[detect-upstream-base](../detect-upstream-base/GUIDE.md).

Confirm v1 with the user before proceeding.

## Step 2: Check for New Upstream Releases

Fetch the latest tags and compare against v1.

```bash
git fetch upstream --tags
```

List upstream release tags newer than v1:

```bash
git tag --list "v*" --sort=-version:refname \
  | while read -r tag; do
      git merge-base --is-ancestor "$V1" "$tag" 2>/dev/null \
        && [ "$tag" != "$V1" ] \
        && echo "$tag"
    done
```

Alternatively, use `gh` for a cleaner check:

```bash
gh release list --repo vllm-project/vllm --limit 100 \
  --json tagName,isPrerelease,isDraft,publishedAt \
  --jq 'map(select(.isPrerelease == false and .isDraft == false)) | sort_by(.publishedAt) | reverse | .[].tagName'
```

Pick the **latest stable release** as the candidate **v2**. Skip pre-release
or release-candidate tags unless the user opts in.

Present to user:
- Current base: **v1**
- Latest upstream release: **v2**
- Release notes link: `https://github.com/vllm-project/vllm/releases/tag/$V2`

If v1 == v2 (already up to date), report that and stop.

Ask: "A new release **$V2** is available. Proceed with rebase?"

Wait for confirmation before proceeding.

## Step 3: Collect Rebase Inputs

Before invoking `rebase-assistant`, gather the remaining required inputs:

1. **Verification checks**: Ask the user which checks to run after rebase.
   Examples:
   - `.venv/bin/python -m pytest tests/path/to/test.py -v`
   - `pre-commit run --all-files`

2. **Reference commit** (optional): A known-good commit where checks pass.
   Defaults to `HEAD` on **b1**.

## Step 4: Rebase

Invoke the `rebase-assistant` skill with the collected inputs:
- **v1**: from Step 1
- **v2**: from Step 2
- **Verification checks**: from Step 3
- **Reference commit**: from Step 3 (if provided)

See [rebase-assistant](../rebase-assistant/GUIDE.md).

Follow the full rebase-assistant workflow (analyze **b1**, back it up, create
**b2**, rebase onto v2, resolve conflicts, verify **b2**).

## Step 5: Summarize and Push

Once rebase-assistant completes, present a final summary:

- **b1** (original branch): preserved as backup
- **b2** (rebased branch): now based on v2
- **Previous base**: v1 → **New base**: v2
- **Custom commits replayed**: count and brief description
- **Conflicts resolved**: list of files and decisions (from rebase-assistant)
- **Verification results**: pass/fail for each check, with log paths
- **Backup branch**: name for rollback to **b1** if needed

Then ask: "Push **b2** to origin?"

If yes:
```bash
git push origin HEAD
```

If the branch already exists on the remote and needs a force push:
```bash
git push origin HEAD --force-with-lease
```

Warn the user before force-pushing and wait for explicit confirmation.

## Usage Examples

```text
Check if there's a new upstream vllm release and rebase onto it.
Checks: pytest tests/entrypoints/openai/correctness/test_transcription_api_correctness.py
```

```text
Auto-rebase to the latest upstream tag. Run pre-commit and transcription tests.
```
