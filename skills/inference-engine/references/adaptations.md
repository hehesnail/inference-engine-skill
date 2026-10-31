# 本地适配与来源保留

固定来源、原文件摘要和本地文件摘要保存在 `source-manifest.json`。完整原始包与测试证据的位置由安装工作区记录。正文保留技术细节，并按可核对的源码与证据修正机制、接口和工具行为；下列各节按时间保留修改范围，当前语义核对见末节。

| 来源 | 文件 | 调整 |
|---|---|---|
| sglang | `.agents/skills/babysit-pr-to-pass-ci/SKILL.md` | 移除自动创建持久 goal 的步骤，授权和重试预算由当前任务决定。 |
| sglang | `.agents/skills/generate-profile/SKILL.md` | 移除按进程名批量清理，设备改为授权变量，模型/门槛为示例。 |
| bbuf | `skills/sglang-humanize-review/SKILL.md` | 保留完整语料查询、风险经验、图示模板和历史说明；将强制全库扫描及图示改为按需。 |

所有嵌套 `SKILL.md` 改名为 `GUIDE.md`；说明中的相对链接指向包内文件。嵌套 AGENTS/CLAUDE 改为 `.reference.md`，只作为对应项目的历史规则参考。原始脚本保留来源摘要；H200 实测与本轮 CPU 反例触及的 helper 已适配，并在当前 manifest 区分原文与修改版本。稳定入口另在 `scripts/offline_tools.py`；收录不自动执行部署、重放或语料更新动作。

许可证：vLLM、vllm-skills 和 SGLang 的 LICENSE 已随各自目录保存。BBuf 此快照未提供根许可证，保留作者、仓库、SHA 和来源链接，作为个人内部参考副本；不将其他仓库的许可证套用到它，也不打包对外发布。

## 本轮离线补齐

- `debug-ima` 增加包内 CUDA core dump 排障入口；保留上游文章出处。
- `model-recipes` 增加本地 YAML/guide 阅读路径，明确源模板与网站已生成命令的差别；在线查询作为可选刷新流程。
- 上游 cookbook 测试仅将 `SKILL_ROOT / "SKILL.md"` 改为 `SKILL_ROOT / "GUIDE.md"`，断言和用例不变；原测试保留在服务器 `expanded/upstream/bbuf/tests/`。

## Astra 指引复查（2026-10-27）

按 [OpenAI 原文](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra) 复查正文，而非仅检查入口长度：

- `babysit-pr-to-pass-ci` 删除残留的自动建 goal、100 行暂停和固定 10 次重试要求；依据当前授权、故障原因及预算推进，保留当前 head、workflow gate 和验证完成条件。
- `cookbook-migrate-model` 将强制回写 skill / 立即追加 commit 改为按维护范围记录已验证、可复用的经验。
- `llm-serving-auto-benchmark` 明确单引擎与所选 workload 的范围；默认校验选中的配置，不要求全框架、全配置执行。
- kernel 对齐、跨路径数值一致性、配置 schema 等真实技术约束保持不变。资料完整性不意味着每次读取或执行全部内容。

## 实际使用复查（2026-10-31）

- 从真实使用中补充三份按需参考：离线生成/评分与整机吞吐口径；自定义模型在启动、调度、缓存、输出与评分中的接线；独立数值基线、约定验收和条件性优化的启用边界。入口保留任务路由，不加载全部资料。
- profiler 参考补充自定义标记、嵌套区域归属、未知项及多 rank 时间范围；保留通用解析器与项目适配器分工，不收录某个模型的私有解析规则。
- 主机、目录、系统限制及复制策略移出通用运行指引。离线入口以相邻 `library/` 为默认，可用参数或环境变量选择资料目录，并列出精简副本的脚本可用性；不自动下载、不启动 GPU 或在线重放。
- 上游指南与脚本保持本轮修改前内容；不将任何具体模型的验收数字、开关阈值或硬件实验结论升级为通用要求。历史调用审计不等同于修订后的模型行为评测。

## 16 仓库深入核对后的吸收（2026-10-31）

新增七份工程参考及来源评估，把 Host/GPU、NCU、kernel 接线、Graph/同步、实验候选、fork 与多后端/PD 分开按需加载。根入口增加 kernel、候选、fork 路由，原有专项内容和稳定脚本不变。

新增资料包保留 TRT 参考/脚本、NCU reference/helpers、NVIDIA sanitizer 参数细节及许可证；NCU 原入口改为 UPSTREAM.reference.md，另写适用范围明确的 GUIDE；TRT 新 GUIDE 标注原脚本的 schema/窗口/rank 假设。相对 Markdown 链接映射到包内文件，未收录依赖指向固定 revision 的源文件，来源摘要留在新 manifest。

没有照搬上游固定收益/精度门槛、默认全卡、latest 镜像、每次写多份计划、缺证据晋升、按名称批量清理和无限迭代要求。也没有因追求抽象通用而删去硬件/算子细节。HF MI355X 的硬件描述以 AMD 官方资料纠正；TRT/其他项目的时间线假设保留为来源限制。

## 完整补齐与事实修正（2026-10-31）

本轮保存 298 项完整技能正文，另补齐 KDA 两个子模块、Atrex 两个知识依赖、cfregly 运行模块/配置/schema、Intel 配套工具、SGLang 多图与扩散路径。只把嵌套技能作为按需参考，不自动注册 MCP、启动 agent、部署服务或运行 GPU。

新增 7 份工程参考，把原工程的输入、状态、模块和结果语义接回推理引擎。原有 vLLM 15 项逐项复核，部署脚本、config diff、KernelAgent success 的边界已 CPU 复现。

KernelWiki 的 PDL/GDC 页按 CUDA 官方说明改写，原文另存 upstream-reference；mbarrier 示例改为逐 stage phase，并区分 arrival/transaction 完成条件。修正 2-CTA 标签，并给 occupancy/stage 经验补上条件。HF MI355X 的架构/缓存混淆、Intel TPOT 与 batch step 的混淆直接在相应指南中标注。硬件内容仍需对应实际架构、版本和数据流。

上游链接按固定源码位置修复；不存在的引用以说明文字保留，并在维护工作区保存缺失清单。重命名和来源标头、链接调整、技术纠错均计入当前 manifest；来源原件在服务器 checkout 保留。上游正文中的固定预算、资源、默认 latest 和强制步骤仍属于原工程场景，不覆盖当前任务授权及验收。

旧 source-manifest 的快照已归档，根 source-manifest.json 表示当前安装版本。完整包与桌面包分别记录当前内容摘要，不能拿旧摘要验证已经修正的文件。

## H200 实测带来的补丁

补充独立 CustomOp 的当前配置上下文、默认 dispatch 验证及改变输入后的 graph replay。实际 trace 暴露的两份 BBuf helper 已作最小修正：DSV3 router 的泛化 gate 关键词收紧；高占比不再升级为 Confirmed；融合结果改为 Name match / Candidate 并说明缺失的调用与依赖证据；混合阶段 all 桶显示为 unclassified。原脚本与实测误报保留在维护证据中，当前文件摘要区分原文和适配版本。相关 26 项回归与真实 trace 回放通过，其他解析功能复用既有证据。

## 逐篇语义核对与能力补强（2026-11-01）

302 份相关 GUIDE、两个非标准工程入口和 KDA 48 份技术页按阅读范围登记，443 个未纳入入口只做元数据筛选。见[能力与范围](semantic-audit.md)、[逐项审阅索引](semantic-review-index.json)。

- 新增六份按需契约说明：attention 后端、量化、speculative、循环状态、Tokenizer/结构化输出、多模态；补齐实际 metadata/mixer、grammar 提交与 rollout 权重版本的数据流。
- Orchestra 33 份正文整理为带适用条件的指南，原正文存入各自 source-notes/guide-at-pin.md；7 份服务参考同步纠正。上游错误示例保留为有标记的历史原件，不再当执行模板。
- 修正 benchmark 成功率/失败数、SLA、有限数值、工作负载一致性与缺失字段；修正 diffusion 缺数据、机械证明失败、schema 依赖缺失与 agent 候选接受的假通过。
- 验证器保留 shape/dtype/键集合与数值类型语义，明确 alias、未返回状态和真实 GPU 未覆盖；修正 Slurm quoting/pipefail、参数生成、host 时间窗与平台跳过状态等相关路径。
- 补回33个被后缀过滤漏掉的资源。原始大语料/媒体保留服务器；两张 VisualGen PNG 仅有 Git LFS 指针，未宣称图像已就绪。原上游签名不能验证适配后的文件。

本轮修订仅更新个人技能包，未向上游提交。真实运行与源码核对分开报告，验证细目见[validation](validation.md)。
