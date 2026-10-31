# vLLM / SGLang 长期 fork 与版本迁移

用于 rebase、上游升级、依赖升级、冲突处理和回归定位。吸收 Cohere 的 fork 四点对照，但用精确 commit 与当前运行环境，避免把最近 tag、最新 wheel 或一次安装模板当成真实基线。

## 四个状态

- `U0`：旧 fork 实际基于的上游提交。
- `F0`：升级前 fork 及其有效补丁/配置。
- `U1`：本次选定的新上游提交。
- `F1`：迁移后的 fork。

先核对 remote URL、当前分支、HEAD、工作树和安装来源。用户指定版本时沿用；否则从仓库关系确认。下面的 `upstream/main` 只是已核对后的例子：

```bash
git status --short
git remote -v
git merge-base HEAD upstream/main
git log --oneline --decorate U0..F0
git diff --stat U0 F0
```

`merge-base` 是共同祖先，不一定等于首次 fork 点；多次 merge/rebase、release 分支和 shallow clone 都会影响解释。`git describe` 的最近 tag 只作标签提示，不能把 tag 当成包含全部实际变更的精确基线；缺历史时按需 fetch 所需范围。

## 保留旧状态后迁移

已有未提交改动应单独保留。备份分支只保存 commit，不保存脏工作树；新 worktree 也不会自动带入未提交修改。按当前仓库和任务选择 patch、临时提交或隔离 worktree，避免丢改动。

对于线性的定制提交，已保存 F0 后，可在新的开发分支/worktree 上使用：

```bash
git worktree add -b NEW_BRANCH NEW_CHECKOUT F0
git -C NEW_CHECKOUT rebase --onto U1 U0
git range-diff U0..F0 U1..NEW_BRANCH
git diff --check
```

替换上述符号为已确认的路径/分支/提交；分支命名沿用工作区约定。merge-heavy fork 需保留适合它的 merge 语义，不能强套线性 rebase，也不默认 squash 所有定制提交。

解决冲突时同时看：旧定制意图 `U0→F0`、上游改变 `U0→U1`、新组合 `U1→F1`。查调用方、状态生命周期和接口变更，尤其是 scheduler/output、KV/state、distributed、custom op schema 与 model weight loader。已由上游实现的功能可以去重，但先确认语义和启用条件对应。

## 环境和验证

- 区分源码迁移和环境迁移。固定镜像复现不先换 latest wheel；C++/CUDA ABI、PyTorch、Triton、FlashInfer、通信库、编译 target 按目标版本实际要求检查。
- editable Python 修改与已编译扩展可能不是同一个版本。核对运行解释器、模块路径、加载共享库和子进程 PATH；测试自己的 checkout，不能误用全局 `vllm` CLI。
- CI test-area YAML 的 working directory、依赖、环境变量与设备数是执行契约。只选相关测试，正确转换 CI 相对路径/shard 参数；不要把 CI 的全部 GPU 或最新安装指令照搬到共享服务器。
- 复用 F0 已有且条件仍有效的通过证据。若 F1 失败，先比较 F0 同一用例；必要时检查 U1 原生是否失败，以区分已有问题、上游回归、定制冲突与环境问题。
- 原有测试通过不能排除组合契约失败。涉及调用方/依赖时，核对实际组合路径，而不是因异常落在依赖库就排除。
- 发布前的测试对象与发布源码要对应；与约定无关的新发现另列观察，不自动扩大迁移目标。

例：上游改变 model runner 的输出对象，旧 fork 的 sampler patch 仍能单测通过，但 scheduler 继续按旧字段消费。修复落在组合路径，验证应进入 runner→sampler→scheduler，而非只重复 sampler 单测。

交付保留 U0/F0/U1/F1、保留/替代的定制语义、相关验证和环境差异。用户仅要求本地迁移时完成本地修复；commit、push、PR 与生产替换按已有授权处理。Cohere 原始流程与差异见 [来源评估](upstream-evaluation.md)。
