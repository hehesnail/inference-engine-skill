# 来源与维护

当前技能名为 `inference-engine`（原 `vllm-sglang`），显示名“推理引擎工作台”。2026-11-01 按 OpenAI 的技能指引复查了触发描述、任务路由与流程约束；已有技术资料及来源版本保留。下文审阅数量与技术验证沿用各自范围。

2026-11-12 从实际跨框架适配、数值定位与缓存准备记录提炼了[数值漂移排查](numerical-drift-debugging.md)，并补充状态、长文容量与 agentic 测量边界。正文保留参考确认、差异定位与证据判断方法；具体算子和兼容办法作为有适用条件的案例，不要求其他模型采用相同 dtype、计算顺序、形状或配置接口。原始版本与实验结果保留在项目证据中。本轮为文档维护，不代表新增 GPU 验收。

当前版本已完成 2026-11-01 的[逐篇语义核对与能力补强](semantic-audit.md)，接续 2026-10-31 的资源补齐。302 份选中正文和 KDA 48 份技术页有逐项阅读、问题与证据记录；历史验证结果仍按各自日期和范围保留，不代表新内容自动通过。

18 个工程的固定 revision、全部 745 个技能条目的采用状态及 302 个相关技能的完整正文，见 [逐仓目录](repository-guides.md) 和根目录 reference-index.json。未采用条目逐项保留理由；KDA、KernelAgent 没有标准 SKILL.md，另有完整工程入口和专项接线。原有官方 vLLM 6+9 项的复核见 [审计](upstream-vllm-audit.md)。

当前文件来源与本包摘要统一记录在 source-manifest.json。它区分原文、适配和本地编写，不把目录复制视作事实验证。具体技术修正见 [adaptations](adaptations.md)，实际运行范围见 [validation](validation.md)。

vLLM 主仓库同步到 7b77ed581d92fa14e34ab43344837bb1ca1b4257；SGLang 同步到 8368f86fd7a71b07e2b7dda5d4146695ec54cbfa，保留全部 33 项，包括多图内存与 5 项扩散模型专项。BBuf 对照 c592955044e896edd20890468c9bdbbe30ab0be6，纳入 11 项相关技能及其配套资料。视频制作入口未采用。其他工程 pin 在逐仓目录和 manifest 中。

KDA 的 KernelWiki 与 NCU 子模块分别固定为 76d27b56f804e7e7295d4c570e1e5d7eef4b0a75 和 d1887948c7d53690cfe6605f59c1329b8a1c6bb5。Atrex 的 KernelWiki 与 GPU 编程教材依赖分别固定为 2777d18ffb3a3d682d8f25a3e3b8864d925a5ff1、1949bde833c8e0f9081f751d99787a63a6526bf1；保存在其完整源码 checkout，不能混作 KDA 的同一份知识库。Atrex 其他参考 kernel/toolchain 子模块按具体任务取用，未声称全部运行。

完整工程、知识语料和原始证据以工作区的服务器存储为准；桌面保留所有选中的修订 GUIDE 及按需细节、相关技术说明和轻量工具。两种安装的范围由 manifest 标记，见 [存储与恢复](server-data.md)。上游明确引用但未随快照提供的文件保留为带说明的路径，不伪造本地文件。

维护时更新受影响专项及其依赖，记录 revision 并验证变化路径；不在日常任务中自动切换 main，也不覆盖复现实验的版本。入口遵循 [OpenAI skills 指引](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra)：简短路由，详细领域知识和可执行工具按需加载，保留真实技术约束，避免把示例工作流升级成固定仪式。
