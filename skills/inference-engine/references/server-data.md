# 完整资料、桌面副本与恢复

安装主机和实际目录由工作区配置决定，不属于通用 skill。完整包与桌面包都保留 302 项选中技能的修订 GUIDE，以及按任务组织的自有说明。桌面副本省略大语料、原始样例 trace 和部分完整工程实现；它们留在服务器，绝不把省略解释成没有相关知识。

完整包另外保留：

- BBuf review 压缩语料与四个引擎的模型 PR 档案。
- KDA KernelWiki 的全部 PR、博客、文档、竞赛材料、代码、patch、PROVENANCE 和候选账本。
- Hyperloom 的工程实现与技术知识目录，以及选中工具的配套运行模块。
- 来源索引、文件摘要与上游许可证。完整原始 checkout、Atrex 两个知识依赖和详细验证产物另存维护工作区。

桌面内的 source-manifest.json 标记每个文件是否安装。Markdown 若指向省略的上游源文件，则转为同 revision 的源链接；日常计算优先使用工作区已保存的服务器副本，不因此自动联网下载整套工程。

稳定 BBuf 工具使用 scripts/offline_tools.py。新增资料检索、wiki 查询和 KDA 导出入口为 scripts/research_tools.py。两者可用 --library-root 或 INFERENCE_SKILL_LIBRARY 选择完整 library。先用 --list 查看脚本与语料状态，再执行所需任务。

KernelWiki 的 INSTALLATION.json 标记 edition。完整查询要求完整文件清单可用；只装技术页时会明确报缺语料。48 页技术正文仍可直接阅读。不要把“脚本存在”当作“全库可查”，也不自动用网络结果填补缺失。

恢复优先使用相同 revision 的服务器归档。大文件不自动传回本机，实际同步范围沿用当前工作区偏好。历史语料的截止日独立于安装日期，也不证明当前 release 已包含其中的修复。
