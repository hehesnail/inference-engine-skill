# 五个 agent 工程如何用于推理引擎

这些工程的价值不只是“计划→实验→总结”。下面按实际模块说明可复用机制、输入输出与接线。每个工程的完整 guides、reference、脚本和模块资料可从 [工程目录](repository-guides.md) 找到。宿主运行时和远端服务依赖仍属于相应工程，不能假装已经安装在本 skill 内。

## KDA：知识检索、候选记录、可交付 kernel

KDA 指 Kernel Design Agents，不是模型里的 Kimi Delta Attention。任务先定义输入/输出、dtype/stride/alias、允许的语言与依赖、正确性命令、计时边界及当前约定的接受条件。已有任务信息足够就执行，无须照搬强制写两份计划的流程。

KernelWiki 提供硬件→技巧→kernel→症状→语言→迁移六类技术页；先用症状/架构查候选，再沿 source ID 找 PR、代码和 PROVENANCE。NCU 提供 metric/PC stall/roofline 解释。知识检索、测量与候选选择是三条相互引用的证据链，不把知识库的 verified 标签当作本机验证。详见 [知识库](kernel-knowledge.md)。

候选保存 parent、源摘要、基线、shape 集、每项正确性和性能证据、接受/拒绝原因。成功 kernel 最后要经过引擎 dispatch 与真实 wrapper，不能只交独立函数。KDA 的 export_huggingface.py 可以把已选 CuTe Python/JSON 候选转成 HF kernel 项目：支持嵌套入口、普通本地 import 改写、run/prepare，拒绝 symlink、动态本地 import 和不支持的源文件；输出目录不能覆盖输入或已有输出。打包副本不改原候选，模块 ID 绑定内容与发布配置。打包成功仍需验证安装后的 run、stream、重复调用和 graph capture，prepare 返回的 launch 时间不等于 run 总时间。原始测试含 fake builder，CPU 通过不证明 GPU 构建或性能。

## CUDA-Agent：把验证器与候选分开

固定 reference/harness，优化 candidate。输入生成、编译、正确性和计时分开；不要通过修改 evaluator、重用旧输出、识别 correctness/timing 阶段或跳过形状换取结果。它的 verify/profile/bench 习惯可用于单算子迭代，固定 1e-2 容差、5 次检查、5% 收益门槛则属于其任务，不能替代当前算子/模型标准。

把 profiler 的 kernel 时间总和与实际调用 wall/event 时间分开。原位 kernel 在重复测量前恢复输入；buffer reuse、L2 状态、dtype、mask 和非连续视图与引擎一致。先通过 reference 对照，再看优化收益。保留失败样例与结果，不只留最快一次。

## KernelAgent：子图优化和组合验证

Fuser/pipeline.py 负责抽取子图、调度候选、汇总，再经 compose_end_to_end.py 拼成完整实现；kernel_perf_agent 提供 NCU、roofline 与历史示例，oink 保留 SM100 的真实算子和 benchmark。

子图边界必须保存输入/输出布局、标量、state/alias/原位副作用以及计算顺序。两个子图单独正确不证明组合正确；融合可能改变累积精度、复用缓存或丢失同步。组合后用原始整图输入验证，再测端到端，核对新增 copy/layout 转换是否吃掉收益。

**已核对的陷阱：** compose 的 verify 默认 False；代码把未验证的生成结果也设为 success=True。必须检查 verify_passed、validator 与日志，不能仅消费 success。该工程的 thread lock 也不是跨进程 GPU 隔离。SM100 特化实现按其 dtype、alignment、stride、in-place 和 fallback 条件使用，不推广为其他架构默认实现。

## Hyperloom：配置、源码、kernel 三条优化路径

实际 Coordinator 维护 session manifest/state/SQLite 事件、baseline、current_best、optimization_stack；FRAMEWORK_AGENT 内配置参数和源码 patch 是两条可独立推进的路径，KERNEL_AGENT 根据 trace 的 reusable_native_kernel_ids 接入 kernel 优化。不要从 README 推断旧版本已删除的 action 仍存在。

可迁移的机制：

- 用完整“源码+镜像/依赖+模型+workload+server args/env+topology”识别实验；配置指纹用于去重，改名字不等于新实验。trace/config 未变时复用解析。
- baseline、当前最好、待测候选分别保存。stack 中记录参数、后端、source、kernel 变化，避免后一轮覆盖前一轮已验证配置。
- source patch 在隔离 checkout 实施。应用、编译/JIT 缓存、实际 loaded module、服务重启构成一项事务；回退需要同时恢复二进制和源码，不能用旧 .so 证明新源码有效。
- resume 读取显式 session id/launch-info，不按“最新目录”猜测。该版本 server args/env 整组替换；多节点还需重新传拓扑和 endpoint handoff。补丁已存在的识别不能代替真实运行。
- quality reference/warm replay 缺失、失败、成功分开。上游部分路径将缺失质量证据视作允许继续；我们的接受条件只按本任务要求，缺失不能报成质量通过。
- breakdown 分离记录的测量与 projected gain、unattributed gain；null/empty 可能是未知。它的 6.1 在同一 major 中改了字段，解析时核对完整 schema 与字段，而不只看主版本。

TraceLens/Magpie/GEAK/KernelForge、KB/PR Monitor 和模型 API 都是外部依赖；保留其安装与运行资料，但只有明确需要实际运行该工程时才配置它们。不会为了借鉴优化逻辑自动部署整套服务或启动多 agent。

## Atrex：不可变测量与独立接受决定

采用 light 路径的 Direction→Experiment→Gateway Record→exact Kernel 关系；legacy 单独保留，不混用两代 API。GPU runtime 在提交时快照源代码、用注入的 evaluator/private inputs 测量；任务 ID 和 kernel ID 用于去重和查询。记录重读成功只证明读到了记录，仍需检查 status、result.correct 和 diagnostics。

每次实验写清假设、可证伪的观察、实际 record id 与解释；解释不能篡改测量。ABBA 把 baseline/candidate 按交错顺序比较，保留每个有效样本及每侧失败；参考产生非有限值不能自动算候选通过。记录缺失、执行中、超时、infra failure、编译失败、numerical failure 分别处置。未知提交结果先查已有 job/record，不盲目重发。

Episode report 被接收是交接，不等于 promotion。最终选中的文件摘要要与证据绑定，不能测完后又改 kernel。runtime-records 与 report-only 两种模式分别使用；后者并不需要伪造 Direction/Experiment ID。移植到普通 SSH 工作流时可以用任务目录的 JSONL 和源快照实现这些关系，不假造 Atrex Supervisor/Gateway API 可用。

micro-analysis、PPU/ACU 联合 profile、timeline、知识检索和 trace mining 的资料保留；PPU/ACU 与专用网关仅在对应平台使用。完整第三方 KernelWiki 与 GPU 编程教材位于服务器源码依赖中；它们与 KDA 的知识库锁定提交不同，不混作一个版本。

真实源码还有以下接入条件：

- KernelAgent 的 runner 仅凭进程退出0和stdout中的PASS/sentinel判成功；这不是可信参考验证器。冻结原任务的oracle、输入和容差，不能让候选打印文本代替执行。LLM router配置也不能关闭任务已要求的整图验证。
- candidate benchmark使用Triton do_bench（warmup/rep为时间预算），PyTorch baseline默认CUDA events（相同数字解释为次数）；源码关键词还可改变dtype。统一计时实现、dtype、cache与state恢复，再计算speedup。原地状态要在每次warmup和trial前恢复；restore是否计时由当前任务固定。
- NCU代码取首个非at kernel而非完整wrapper，aggregate Memory SOL也不等于DRAM吞吐。保存全部相关launch及源摘要；95% SOL不能作为引擎端到端已最优的证明。
- Hyperloom quantization的eval driver默认score越高越好，ppl10→20可被算成零损失；metric方向、source/quant score及阈值的有限性必须来自任务契约，不能交由LLM仅填relative_gap。
- Atrex wiki-gate缺jsonschema时只做弱校验，不能把这种依赖缺失分支标为schema通过。已经有jsonschema时保留正常校验。

cfregly 的部署注册表是候选workaround清单：当前实现按model名查找并regex扫描全文，注释和旧配置也可能匹配；必须核对engine/source、硬件、模型revision和实际resolved server配置。容量工具的infeasible SLA行不能因给出有限pods就当作满足SLO。结构化baseline diff只比较共同有限字段，缺失必需字段另作错误。

serve sweep按本次attempt保存原始输出；非零退出不能晋升为full，requested concurrency必须全部存在。TTFT mean与median分列，聚合output/concurrency不能冒称每个用户的decode速度；未观测的graph模式保持unknown。冷/暖缓存、启动/重启及源码实际加载身份在baseline/current-best/stack中一起记录。
