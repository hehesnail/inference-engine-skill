> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[NVIDIA/TensorRT-LLM@f405e10b730e](https://github.com/NVIDIA/TensorRT-LLM/blob/f405e10b730ea953961ecf5719242c581527587a/.claude/skills/trtllm-flashinfer-upgrade/SKILL.md)，2026-10-31 核对。

# FlashInfer Version Upgrade Skill

Automates upgrading the `flashinfer-python` package version across TensorRT-LLM.

## When to Use

- User asks to upgrade / bump / update flashinfer
- Routine dependency update duty for flashinfer-python

## Local change versus publication

A local dependency update can proceed with the existing target version and base checkout without gh authentication or a fork. Steps 0a–0c and Step 7 apply only when publication is requested; use existing session identity, transport and permissions. Do not repeat version-choice questions when the user already supplied a target, or install/upgrade the shared environment merely to match this template.

## Prerequisites for authorized publication

### Step 0a: Determine GitHub Username

Query `gh` for the authenticated user's login:
```bash
GITHUB_USERNAME=$(gh api user --jq .login)
echo "$GITHUB_USERNAME"
```

If this fails, `gh` is not authenticated — resolve Step 0c first, then retry.
As a fallback, derive the username from the fork remote:
```bash
GITHUB_USERNAME=$(git remote -v | grep -E 'github\.com/[^/]+/TensorRT-LLM' \
  | head -1 | sed -E 's|.*github\.com[:/]([^/]+)/TensorRT-LLM.*|\1|')
```
If neither works, ask the user via `AskUserQuestion`.

### Step 0b: Verify Fork Remote

Check that a git remote pointing to the user's fork of TensorRT-LLM exists:

```bash
git remote -v | grep -E "github\.com[:/]${GITHUB_USERNAME}/TensorRT-LLM(\.git)?([[:space:]]|$)"
```

If **no fork remote** is found, continue the local update; resolve the intended fork only when publication requires it:

> No GitHub fork remote detected. A fork of `NVIDIA/TensorRT-LLM` is required
> to push branches and create PRs.
>
> 1. Fork the repo at https://github.com/NVIDIA/TensorRT-LLM/fork
> 2. Add it as a git remote:
>    ```bash
>    git remote add fork https://github.com/<GITHUB_USERNAME>/TensorRT-LLM.git
>    ```
> 3. Verify the fork destination before the authorized publication step; local preparation can continue.

### Step 0c: Verify `gh` CLI Is Authenticated

The publication example uses `git push` for Git transport and the GitHub CLI (`gh`) for PR creation. Confirm gh is
installed and authenticated:

```bash
gh auth status
```

Authentication does not prove repository permission. Verify the intended account and repository access using the existing credential method; Git SSH credentials and gh API credentials may differ. Do not require a broad classic `repo` scope when existing fine-grained or app permissions suffice.

If `gh` reports "not logged in", instruct the user:

> ```bash
> gh auth login
> ```
>
> Choose: GitHub.com → HTTPS → authenticate with a web browser (or paste a PAT
> with `repo` scope).

**Note on `GH_CONFIG_DIR`:** If the user keeps multiple `gh` accounts (e.g. a
personal account and a separate account for `NVIDIA/TensorRT-LLM` work), they may
point `gh` at a non-default config directory. Check `CLAUDE.local.md` /
`AGENTS.md` or the environment for `GH_CONFIG_DIR`; if unclear, ask the user.
When set, prefix every `gh` invocation: `GH_CONFIG_DIR=<path> gh ...`.

Do not publish until the intended destination and required access are verified. Missing gh/fork setup does not block inspecting pins, preparing a local patch or running authorized isolated checks.

## Workflow

Execute these steps **in order**. Use `AskUserQuestion` for user choices and
`WebFetch` / GitHub API for release data.

### Step 1: Fetch Available Releases from GitHub

Fetch the release list from `https://github.com/flashinfer-ai/flashinfer/releases`.

Use `WebFetch` with the URL `https://github.com/flashinfer-ai/flashinfer/releases`
and extract all release tag names and dates. Collect both stable releases
(e.g., `v0.6.7`) and pre-release / nightly tags (e.g., `v0.7.0.dev20260401`).

Alternatively, use the GitHub API via curl:
```bash
curl -s "https://api.github.com/repos/flashinfer-ai/flashinfer/releases?per_page=30" \
  | python3 -c "
import json, sys
releases = json.load(sys.stdin)
for r in releases:
    tag = r['tag_name']
    pre = ' (pre-release)' if r['prerelease'] else ' (stable)'
    date = r['published_at'][:10]
    print(f'{tag}  {date}{pre}')
"
```

### Step 2: Check Current Version

Read the current pinned version from `requirements.txt`:
```bash
grep flashinfer-python requirements.txt
```
Expected format: `flashinfer-python==X.Y.Z`

### Step 3: Ask User Preferences

Resolve the following choices from the task/session first; ask only for missing decisions that materially affect the update:

1. **"Prefer a latest nightly release version?"**
   - Options: "Yes, show nightly/dev releases" | "No, stable releases only (Recommended)"
   - This filters the release list shown in the next question.

2. **"Which flashinfer-python version do you want to upgrade to?"**
   - Present up to 4 versions newer than the current version (filtered by
     the nightly preference above), with the latest as the recommended option.
   - If the current version is already the latest, inform the user and stop.

3. **"Also update `security_scanning/poetry.lock`?"**
   - Options: "No, skip the lockfile (Recommended)" | "Yes, update version + hashes"
   - Default: **No**. The lockfile is typically regenerated by maintainers
     separately; editing it here can produce spurious hash diffs and stale
     `metadata.content-hash` values.
   - If the user answers **Yes**, follow the "Updating `security_scanning/poetry.lock`
     hashes" subsection below; otherwise skip it entirely (do not touch
     `security_scanning/poetry.lock`).

### Step 4: Update All Version References

After the user selects a target version, update these files:

| File | What to change | Always |
|------|----------------|--------|
| `requirements.txt` | `flashinfer-python==OLD` → `flashinfer-python==NEW` | Yes |
| `security_scanning/pyproject.toml` | `"flashinfer-python (==OLD)"` → `"flashinfer-python (==NEW)"` | Yes |
| `ATTRIBUTIONS-Python.md` | `## flashinfer-python (OLD)` → `## flashinfer-python (NEW)` | Yes |
| `security_scanning/poetry.lock` | Update `version = "OLD"` → `version = "NEW"` under `[[package]] name = "flashinfer-python"`, and update the `files` list with new hashes | Only if user opted in at Step 3 question 3 |

#### Updating `security_scanning/poetry.lock` hashes

> Only perform this subsection if the user answered **Yes** to question 3 in
> Step 3. Otherwise skip it entirely.

The poetry.lock file contains SHA256 hashes for the wheel and sdist. Fetch them
from PyPI:
```bash
curl -s "https://pypi.org/pypi/flashinfer-python/NEW_VERSION/json" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
for f in data['urls']:
    print(f'{f[\"filename\"]}  sha256:{f[\"digests\"][\"sha256\"]}')
"
```

Replace the old `files = [...]` block under `[[package]] name = "flashinfer-python"`
with the new filenames and hashes. Also update the `[package.dependencies]` section
if the new version has different dependencies (check PyPI JSON `requires_dist`).

**Important**: After manually editing both `security_scanning/pyproject.toml` and
`security_scanning/poetry.lock`, the lockfile's `metadata.content-hash` becomes stale.
Regenerate it by running:

```bash
cd security_scanning
poetry --version
poetry lock --help
# Poetry 2.x preserves existing locked versions by default:
poetry lock
# Older versions may expose --no-update; use only when listed by this version.
```

Resolve with the repository's supported Poetry version and review the complete lock diff; a changed dependency may require transitive changes. Do not promise that only a content hash changes. If `poetry` is
available, you can alternatively use `poetry add flashinfer-python@NEW_VERSION` in the
`security_scanning/` directory to update both `pyproject.toml` and `poetry.lock`
automatically (including the content-hash).

#### Nightly / dev version special handling

If the user selects a nightly/dev version (e.g., `0.7.0.dev20260401`; historical example, not a verified available release):
- The PyPI package may not exist — check first with `curl -s "https://pypi.org/pypi/flashinfer-python/VERSION/json"`.
- If not on PyPI, the `security_scanning/poetry.lock` hashes cannot be updated.
  Record the unresolved artifact/source and keep the lockfile coherent with its declared source. A TODO comment cannot validate missing distribution hashes or make a stale lock current.
- The `requirements.txt` can pin to a git install instead:
  `flashinfer-python @ git+https://github.com/flashinfer-ai/flashinfer.git@TAG#egg=flashinfer-python`
  Ask the user which approach they prefer (PyPI pin vs git pin).

### Step 5: Verify Version Compatibility

After updating, check if any code has version-gated logic that needs adjusting:

```bash
grep -rn 'flashinfer.*__version__\|flashinfer.*version' \
  tensorrt_llm/ --include="*.py"
```

Known locations with version checks:
- At f405e10b730e, `tensorrt_llm/_torch/speculative/interface.py` rejects an unavailable FlashInfer or `Version(flashinfer.__version__) < Version("0.6.4")`; this is a semantic-version check, not a lexical string comparison.

A satisfied version gate is only one check. Inspect changed FlashInfer signatures, supported dtypes/shapes, kernel layouts and JIT/CUDA dependency compatibility in affected call sites, then run the relevant isolated import/CPU checks and authorized runtime tests. Compare versions with a semantic version parser, not lexical string ordering; record unrun paths.

### Step 6: Summary

Print a summary of all changes made:
- Old version → New version
- Files modified (with line numbers)
- Any warnings (e.g., poetry.lock hashes couldn't be updated for nightly)
- Validate installation in the task's isolated server environment or target image; preserve the shared environment and record resolved versions
- Remind user to run relevant unit tests:
  ```bash
  pytest tests/unittest/_torch/flashinfer/ -v
  pytest tests/unittest/_torch/attention/test_flashinfer_attention.py -v
  ```

### Step 7: Commit, Push, and Create PR

After all files are updated and verified:

> If the user opted **out** of the `poetry.lock` update at Step 3 question 3,
> drop `security_scanning/poetry.lock` from the `git stash`, `git add`, and
> commit message in the snippets below.

#### 7a. Preserve the task base and existing changes

Only for authorized publication. Check HEAD, remotes and the worktree first. Use an isolated worktree or a recovery point; do not move unrelated changes or silently replace the agreed base with latest main. The following is an upstream-main example only when that base was selected:

```bash
# Drop security_scanning/poetry.lock from this list if the user opted out.
git stash push -m "flashinfer-upgrade-wip" -- requirements.txt security_scanning/pyproject.toml security_scanning/poetry.lock ATTRIBUTIONS-Python.md
git checkout main
git pull --rebase https://github.com/NVIDIA/TensorRT-LLM.git main
git checkout -b ${GITHUB_USERNAME}/update_flashinfer_${NEW_VERSION}
git stash pop
```

Where `GITHUB_USERNAME` comes from the fork remote (e.g., `yihwang-nv`) and
`NEW_VERSION` is the selected version (e.g., `0.6.7.post3`).

#### 7b. Commit with DCO sign-off

```bash
# Drop security_scanning/poetry.lock from the `git add` list and the commit
# body if the user opted out.
git add requirements.txt security_scanning/pyproject.toml security_scanning/poetry.lock ATTRIBUTIONS-Python.md
git commit -s -m "[None][chore] Update flashinfer-python from OLD to NEW

Bump flashinfer-python dependency to the latest stable release.
Updated version pins in requirements.txt, security_scanning/pyproject.toml,
security_scanning/poetry.lock (if updated), and ATTRIBUTIONS-Python.md."
```

#### 7c. Push the branch to the user's fork

Identify the fork remote (from Step 0b — commonly named `fork`), then push:

```bash
FORK_REMOTE=fork   # adjust if the user named their fork remote differently
BRANCH="${GITHUB_USERNAME}/update_flashinfer_${NEW_VERSION}"
git push -u "${FORK_REMOTE}" "${BRANCH}"
```

If a push is rejected, distinguish Git transport credentials, gh authentication and repository permissions. A gh login does not necessarily configure the Git credential helper or SSH identity. Preserve the established credential method; do not expose tokens in commands or logs.

#### 7d. Open the PR on `NVIDIA/TensorRT-LLM`

```bash
# Write the exact multiline PR body to "$WORK_DIR/pr-body.md" first, then:
gh pr create \
  --repo NVIDIA/TensorRT-LLM \
  --base main \
  --head "${GITHUB_USERNAME}:${BRANCH}" \
  --title "[None][chore] Update flashinfer-python from ${OLD_VERSION} to ${NEW_VERSION}" \
  --body-file "$WORK_DIR/pr-body.md"
```

`gh pr create` prints the new PR URL on success. Report it back to the user.

## Files Reference

All files that contain flashinfer-python version pins:

| File | Pattern |
|------|---------|
| `requirements.txt` | `flashinfer-python==X.Y.Z` |
| `security_scanning/pyproject.toml` | `"flashinfer-python (==X.Y.Z)"` |
| `security_scanning/poetry.lock` | `name = "flashinfer-python"` block with version + hashes |
| `ATTRIBUTIONS-Python.md` | `## flashinfer-python (X.Y.Z)` |

## Notes

- The `setup.py` has a comment about git+https install URLs — no version pin to update there.
- The `.pre-commit-config.yaml` and `pyproject.toml` reference flashinfer source files, not versions — no changes needed.
- The `flashinfer/` submodule (if present) is separate from the `flashinfer-python` PyPI package.
