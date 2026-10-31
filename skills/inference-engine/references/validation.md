# 验证记录

当前版本见本页末节的 2026-11-01 语义核对；较早各节保留当时的文件数量、代码与运行范围，不能直接相加。

2026-10-27 在 Linux CPU 环境完成；未启动模型服务、GPU 推理或性能实验。

- BBuf 10 个测试文件：**190 passed，283 subtests passed**。覆盖容量、profiler、benchmark 对比、模型配置、layer track、轨道压缩、cookbook、incident 产物/重放逻辑与 review 查询。测试中的 HTTP 行为使用 mock 或本机临时服务，不代表真实推理服务已通过。
- cookbook 的一个测试仅将文档路径 `SKILL.md` 适配为 `GUIDE.md`，断言不变。路径适配详情见 `adaptations.md`；原始上游测试文件另存服务器。
- 15 个离线工具全部通过 `--help`；列表、缺少输入及在线采集参数拒绝行为通过。
- 通过统一入口实际处理合成 trace 和 vLLM 启动日志；识别出两个 kernel 的 100/60 微秒耗时，以及日志中的 14.5 GiB 权重、52.25 GiB KV 池和 1.25 GiB CUDA graph。合成数据仅验证分析管线，不是性能结果。
- 用已保存语料实际执行 review 关键词查询和模型 PR 历史查询，返回可追溯记录；不依赖上游即时下载。
- 99 个 Python 文件完成 AST 语法检查；8 个 Shell 脚本完成 `bash -n`。这不是对所有脚本、部署环境或 GPU 功能的运行验证。
- 顶层 skill 使用系统 `quick_validate.py` 校验；桌面包内 Markdown 相对链接、唯一技能入口和 source-manifest 摘要在安装前校验，详细计数见服务器的 `final-checks.json`。

Python CLI 检查使用服务器 Python 3.10；pytest 使用已有 tokenizer-audit 环境，并通过 PYTHONPATH 复用服务器现有 requests 与 PyYAML，未安装/升级依赖。

原始证据位置由安装工作区的维护记录保存：`offline.log/xml` 为回归结果，`entrypoints.json` 及各命令日志为入口验证，`final-checks.json` 为包检查。本包保留验证范围说明，原始日志和测试夹具按工作区存储策略保留。

同日按 Astra 文章进行了人工语义复查，修正详见 `adaptations.md`。本轮仅改变三份 GUIDE 的流程措辞；脚本保持不变，复查包内链接、来源摘要及 skill 格式。上述 190 项测试证明工具和相关逻辑的可运行性，不是模型遵循指令、正确选资料或减少无效工作的行为评测；后者仍需用真实任务持续检验。

## 使用审计后的修订验证（2026-10-31）

在 Linux CPU 环境完成 14 项入口测试：带空格路径下的迁移、相邻 library 默认值、环境变量与显式参数优先级、精简包工具列表、缺失脚本/语料提示、显式 corpus、仅安装 SGLang 历史目录、子进程失败传播、CPU 环境与离线参数限制。平台分支用模拟检查确认不再禁止 Windows；没有运行 Windows 原生集成测试。

同一已保存合成 trace 经新旧入口处理，标准输出一致。此轮未修改上游解析器，复用之前回归证据，没有重新执行全部上游测试或模型/GPU 实验。当时根入口为 41 行；自有入口与参考的 115 个相对链接可解析，skill 格式、wrapper AST 和个人主机/路径残留检查通过。

实际使用审计覆盖两条聊天的 127 轮记录，其中 42 轮存在可见的 skill 相关读取、复制或工具命令；另核对 8 份已保存 profiler 输出。它说明旧版在哪些任务中有用、出现过哪些执行偏差，不证明修订版在所有新任务中都会被正确遵循。逐轮索引、具体案例和测试日志由维护工作区保存，不打包用户任务、服务器设置或模型验收数字。


## 上游源码核对与工程补强验证（2026-10-31）

上一轮根入口为 46 行，原有 52 份 GUIDE 保留，新增 TRT Host/Graph、NCU 和 Compute Sanitizer 三个按需资料入口，共 55 份 GUIDE；另新增七份工程参考及逐仓来源评估。

- 系统 quick_validate 通过；新增/修改 Markdown 的相对链接检查通过。
- 59 个新来源文件的摘要与 source-manifest-20261031.json 一致；保留各自许可证。
- 7 个新增上游 Python 文件完成 AST 语法检查；这不是 helper 运行验证。
- 用 SQLite 内存夹具实际调用 TRT 上游 Host 函数：跨窗口样例返回 0.20 µs，正确裁剪并集为 0.15 µs；双设备样例返回合并的 0.20 µs，而目标设备为 0.10 µs。复现支持本包关于原脚本窗口/设备假设的限制说明，不是推理性能数据。
- 当前 CPU 维护环境没有 ncu / ncu_report，因此新增 NCU helpers 尚未用真实报告运行；未注册为已验证的稳定离线工具。GPU 采集、模型/数值回归、性能实验和外部 agent 运行均未执行。
- 原稳定 wrapper 与旧解析器没有改变，复用原回归证据，未重复运行全部旧测试。此次内容/来源检查不等于独立 agent 行为评测；实际任务中仍应按证据修正。

完整源码 checkout、阅读索引、CPU 复现结果和安装校验记录由维护工作区保存；不把服务器地址或固定硬件实验阈值写入此技能。


## 完整工程补齐后的阶段验证（2026-10-31）

本节覆盖本轮新增或修正的路径，不能与前几节数字相加当作单次全量回归。

- 12 项 research_tools 测试通过：固定来源检索、架构别名、source→code、完整/残缺语料、外部环境变量、错误传播及全部选中 GUIDE 可达。
- cfregly 的 8 组证据、基线、候选选择、capture signature、verdict、spec metrics 和 roofline 测试：114 passed，无 skip。使用本包实际运行模块；补齐 CPU 的 pyarrow/matplotlib 后重跑了此前被跳过的用例。
- KDA 导出测试使用本包脚本：32 passed、1 skipped。跳过项需要真实 kernel-builder；当前环境未安装该 Rust/Nix 工具链。通过的用例含 fake builder，不能宣称真实构建、CUDA 编译或 GPU 运行已通过。
- 修正后的 KernelWiki：2265 页、2217 个 source ID、89 组代码资料（64 verbatim / 13 extracted / 12 derived）、6 份候选账本的 schema/引用检查通过；5 项 DoD fixture 通过。这是结构校验，不是对 2265 页事实逐条背书。
- CPU 行为复现：真实 HTTP 500 下 curl -s 返回 0，curl -fsS 返回 22；实际 diff_dumps.py 漏报 after-only 条目、error 仍返回 0；实际 compose 函数在模型响应 stub、禁止调用验证器的条件下 verify=False 仍返回 success=True。三个复现均非模型/GPU 实验。
- 当前完整包和桌面包分别校验技能格式、相对链接、唯一入口、选中资源和文件摘要；详细安装校验保存在维护工作区。
- 入口 48 行。全部 745 个技能条目有采用状态，298 项选中正文可读；全文收录、源码路径审阅、事实核验和实际运行分别记录，不能混称全部实测。

CPU 阶段共 158 项上述 pytest 用例通过、1 项真实 builder 用例跳过；DoD fixture 与包结构检查单列。该 CPU 阶段没有运行 GPU、完整外部 agent/MCP 服务、集群部署或跨硬件性能实验，也没有执行独立模型行为评测。未来实际会话的错误路由、遗漏读取或误用阈值仍需用任务证据修订。

## 授权后的 H200 实际流程验证（2026-10-31）

随后按用户授权，在一张 H200 上完成真实运行，其他卡未使用。使用已有 vLLM 0.29.0+cu129 / PyTorch 2.13.0+cu129 环境、冻结的当前模型适配代码和既有权重，不修改原 checkout 或升级环境。用户模型、主机、目录及原始 trace 仅保留在工作区报告中。

- 离线 B2：输入为已有计划的两条真实请求；预热后独立计时两轮，每轮实际生成 64 token，输出逐 token 一致。另采 4 步 decode，profile 请求的生成前缀也与计时运行一致。这个结果只证明该路径可运行、测量口径和重复一致性，不是新模型质量验收或优化收益。
- 实际 vLLM SiluAndMul：FP16/BF16 × B1/B8/B32，共 6 组。正常 dispatcher 的 CUDA 路径、eager、capture、改变输入后的 replay 均与 FP32 中间计算的参考比较通过；明确保存本用例容差，未覆盖所有布局和硬件。
- 首次独立 kernel 脚本因缺少 VllmConfig 上下文失败。按安装包源码修正配置和 dispatch 验证后通过，经验已补入 kernel-engine-integration；失败记录保留。
- 完成后的单次资源确认显示所用 GPU 显存占用回到 0 MiB。CPU 分析在开发服务器进行，未把原始 trace 下载到桌面。
- 本包 profile 工具实际读取了模型 trace 的 24,250 个 GPU 事件与 kernel trace 的 8 次 GPU launch。发现 GDN 被误匹配为 DSV3 router、把耗时占比当 Confirmed、混合阶段 all 标签易误读；修正两份 helper 后，以同一真实 trace 回放确认消除误报。
- 相关上游与新增回归共 26 passed、14 subtests passed。新增 4 项分别约束 GDN 误配、占比不能确认依赖、名称命中不能确认 dispatch、未归类桶不能显示为合计。

本轮 CPU pytest 总计 184 项通过、1 项真实 kernel-builder 构建用例跳过；另列 5 项 DoD fixture 与 6 组 GPU kernel 用例。此次没有做 SGLang 实跑、其他后端性能验证或独立新 agent 的行为评分，不能推广为整套知识库都经过硬件验证。

## 逐篇语义核对后的验证（2026-11-01）

本轮针对实际修改路径在 Linux CPU 环境运行；使用实际待安装包，个别硬件/子进程依赖以明确的 stub 或局部函数抽取验证。没有新增 GPU、真实服务、Slurm payload 或外部 agent 全流程运行。

| 路径 | 当前结果与边界 |
|---|---|
| benchmark 结果比较 | 15 项 pytest 通过；有限数值、实际成功率/SLA、缺 p50、配置不匹配、失败计数与科学计数法 |
| agent 候选/质量判断 | 19 项 pytest 通过；sweep 失败/空数据、质量方向与基线、schema 缺依赖等 |
| research_tools | 12 项 pytest 通过；当前302份 GUIDE 可达与资料/错误传播 |
| cfregly 8组 CPU 模块 | 120 项 pytest 通过；本轮实际模块的证据/基线/选择/指标回归 |
| Slurm 环境探测 | 5 项 pytest 通过；真实 CPU shell/subprocess，未向集群提交作业 |
| 平台与硬件 helper | 27 项 CPU 检查＋7 项 C++ mock-CPUID 检查通过；不是目标 GPU 后端执行 |
| TRT 平台/验证器 | 37＋24 项 CPU 检查通过；真实 parser/comparator、stub kernel 响应与 CPU 子进程 |
| TRT 配置/命令/报告 | 17＋4 项 CPU 检查通过；参数、quoting、默认值、pytest 计数与 CLI 边界 |
| diffusion 指标 | 6 项 CPU 检查通过；缺失、格式错误、非有限数值不得通过 |
| 机械重构证明 | 无效 Git object 触发 RuntimeError；不再把失败命令当成功证明 |
| 生成器配套模板 | 合成 recipe 实际生成5个文件，3份 shell 通过 bash -n；未执行生成的部署命令 |
| 语法与结构 | 本轮33个修改/补回脚本 AST 或 bash -n 通过；skill格式、入口与包内链接另行核对 |
| KernelWiki | 2265页、2217 source ID、89代码包、6账本结构检查及5个DoD fixture通过；不是2265页事实背书 |

这些检查分组记录，不与10月31日的158/184项历史 pytest、相互包含的用例或重复运行次数相加。未改动的 KDA 导出沿用此前32通过、1个真实 kernel-builder 跳过；不能把 fake builder 解释成 Rust/Nix/CUDA 构建通过。

本轮全部302个选中 GUIDE 正文、KDA48技术页与两个非标准工程入口均有审阅记录。关键实现或配套资料仅按逐项记录的范围核对，不宣称所有源码、每份附属文件和完整 PR 历史逐行读完。443个未采用入口只完成元数据筛选。

从49行根入口独立复核三个实际任务方向：混合Mamba状态、70B量化容量与成功吞吐、JSON流式/speculative状态。补上 metadata/mixer 和 grammar 提交源码入口。这是资料路由和分析契约评估，不是这些故障的实跑证明。

H200结果仅按上一节范围复用。当前没有覆盖 SGLang 真实运行、ROCm/XPU/Blackwell/Jetson等目标后端、真实集群和全部外部 agent。包与安装摘要只证明交付一致，不证明未来任务必然正确。完整阅读、补丁、CPU日志与安装证据由维护工作区保管。
