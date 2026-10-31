> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[cohere-ai/vllm-skills@2a9ed1ff5dd8](https://github.com/cohere-ai/vllm-skills/blob/2a9ed1ff5dd822a521a453f49eff84e8dd5a7071/skills/detect-upstream-base/SKILL.md)，2026-10-31 核对。

# Detect Upstream Base Tag

Determine which upstream vllm tag the current branch is forked from.

## Usage

### Step 1: Ensure upstream remote

```bash
if ! git remote get-url upstream &>/dev/null; then
  git remote add upstream git@github.com:vllm-project/vllm.git
fi
git fetch upstream --tags
```

### Step 2: Detect v1

```bash
V1_BASE=$(git merge-base HEAD upstream/main)
V1=$(git describe --tags --abbrev=0 --match="v*" "$V1_BASE" 2>/dev/null)
```

If `git describe` fails (no reachable tag), fall back to the merge-base SHA:
```bash
V1=${V1:-$V1_BASE}
```

### Step 3: Confirm with user

Present to user: "The current upstream base tag appears to be **$V1**. Is
this correct, or would you like to specify a different tag?"

Wait for explicit confirmation before returning the value.

## Output

The confirmed value of `V1` (an upstream tag like `v0.19.1` or a commit SHA).
