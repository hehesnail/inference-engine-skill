# 推理引擎工作台 · inference-engine

为 vLLM、SGLang 及相关推理 runtime 的开发、排障、部署、数值验证、性能分析和 kernel 优化服务。入口保持简短；细节按问题从 references 和 library 加载。

这是个人维护的**私有资料与技能仓库**。完整能力与来源保存在这里，安装时按需要选择文件；不绑定某台服务器、GPU 编号、模型路径或 SSH 设置。

## 换电脑安装

需要 Codex、Git、Python 3.9+。下面的一行安装入口还需要 GitHub CLI `gh`。私有仓库先在新电脑执行一次 `gh auth login`，登录有权访问本仓库的账号。CLI 登录与 Codex 的 GitHub 连接相互独立，不需要把 token 写进脚本。

Windows PowerShell：

```powershell
gh api -H 'Accept: application/vnd.github.raw+json' repos/hehesnail/inference-engine-skill/contents/install.py | python - --update
```

Linux / macOS：

```bash
gh api -H 'Accept: application/vnd.github.raw+json' repos/hehesnail/inference-engine-skill/contents/install.py | python3 - --update
```

同一条命令会安装或更新 `inference-engine`、`lmdeploy-pipeline` 和 `pytorch-refactor`。三个同名目录都位于 `$CODEX_HOME/skills/`，未设置 `CODEX_HOME` 时使用 `~/.codex/skills/`。安装器用当前 GitHub CLI 登录读取私有仓库，默认下载通用技能的桌面版及两个完整的 LMDeploy 专项技能。

只有旧版 `inference-engine` 的机器重跑命令，就会补装两个专项技能。安装器复用内容相同的技能目录，更新有差异的目录前会将旧目录整体移到同一父目录下的 `skill-backups/`。重复安装相同版本不会增加备份。用 `--check` 可以只校验三个技能的安装内容。

Windows 上若通过 `py -3` 启动 Python，将命令里的 `python` 换成 `py -3`。安装后可在 Codex 下一轮对话中使用三个技能。

## LMDeploy 专项技能

- [lmdeploy-pipeline](skills/lmdeploy-pipeline/SKILL.md) 用于 Python pipeline、自托管服务和客户端验证。
- [pytorch-refactor](skills/pytorch-refactor/SKILL.md) 用于 LMDeploy PyTorch 引擎的模型接入、算子分派和运行时排障。

## 完整版、已有 checkout 和回退

完整版包含服务器使用的较大知识语料、代码证据和历史资料，推荐装在服务器。给上面命令末尾加 `--edition complete` 即可。`--edition` 选择通用技能的版本。两个专项技能在两种版本中使用同一组文件。具体数量以 `distribution.json` 为准。

已有 Git checkout 或使用 SSH 凭据时：

```bash
git clone git@github.com:hehesnail/inference-engine-skill.git
python3 inference-engine-skill/install.py --update
```

这会克隆完整维护仓库，适合服务器维护；日常笔记本优先使用前面按需下载的一行命令。安装器可通过 `--repo git@github.com:hehesnail/inference-engine-skill.git` 使用已有 SSH 身份；默认 HTTPS + gh 登录。可用 `--dest /path/to/skills` 指定安装父目录，`--ref <tag>` 固定远端版本。通过 stdin 运行安装器时才需要 --repo/--ref；从 checkout 运行时安装当前 checkout 内容。

回退可安装旧 tag（配合 `--update`），或先将当前技能目录移出 skills，再把 `skill-backups/` 中所需旧版本移回 `skills/` 下对应的同名目录。备份不自动删除；确认不需要时自行清理。

## 内容

- 短入口、按问题路由的 references、302 个已选上游指南及相关模板、脚本。
- vLLM / SGLang 源码与服务、模型接入、状态与 KV cache、Tokenizer / 输出契约、量化与精度。
- Benchmark、公平对照、Torch / Nsys / NCU 分析、显存与容量、kernel 接线、Graph / 同步、并行与通信。
- KDA / KernelWiki 技术知识、agent 优化工作流、历史 review 与版本迁移工具。
- `scripts/offline_tools.py` 和 `scripts/research_tools.py`：CPU 离线入口；依赖按子命令安装，安装技能本身不安装 CUDA、推理框架或模型。

**安装成功只代表技能资源已安装，不代表推理运行环境或 GPU 实验已验证。** GPU、服务器与资源窗口使用任务和工作区提供的配置。

## 如何维护

通用技能 `inference-engine` 的完整维护源是 `skills/inference-engine/`。`profiles/desktop/` 只保留桌面版需要替换的少量文件，例如远端资源链接、安装状态；不是另一份独立技能。

1. 在服务器 checkout 中修改对应的 `skills/<技能名>/` 目录。两个专项技能的入口、参考和样例也会由下方脚本生成文件清单与校验值。
2. 新增文件默认进入两种安装版本；只应在服务器保留的大文件，在 source-manifest 对应记录中设置 `placement: server-only`。
3. 修改有桌面覆盖的文件时同步检查 `profiles/desktop/`，工具会阻止忽略这个步骤；检查后使用 `--accept-overrides` 刷新。
4. 运行：

```bash
python3 tools/refresh_distribution.py
python3 tools/refresh_distribution.py --check
python3 -m unittest discover -s tests -v
python3 install.py --source . --dest /tmp/inference-skill-check/skills
```

5. 核对 diff、提交并推送 `main`；新电脑或已有电脑重跑安装命令即可取到更新。可打 tag 保留稳定版本。

`source-manifest.json` 保留上游 repo/revision、来源路径、修改状态和每个文件摘要。`distribution.json` 是安装器读取的派生索引，文件改变后必须重建。安装前和复制后都会核对摘要；内容损坏时不会动现有安装。

上游项目自己的指令和脚本是按需参考资料；不会在安装过程中执行。第三方权利说明见 [NOTICE.md](NOTICE.md)。
