# 推理引擎能力的语义核对

审阅起于 2026-10-31，修订与安装收尾于 2026-11-01。重点是把源码、状态与证据接回实际推理任务。根入口保持49行，领域细节分层读取。

## 范围与证据层次

18个主工程的745个标准技能入口保留来源与选择状态；本轮采用302项，逐篇读取正文并记录保留能力、问题、适用条件和修订落点。另读KDA/KernelAgent工程入口、KDA的48篇技术正文及相关实现。对应的完整PR历史、所有仓库源码和每一份附属资源没有逐行审阅，443个未采用入口只做名称/描述范围筛选，不称为无用。

[逐项审阅索引](semantic-review-index.json) 对应具体GUIDE、固定SHA与证据范围。[工程目录](repository-guides.md) 和根 `reference-index.json` 用于检索。源码目录、下载/测试原件和所有阅读记录由维护工作区保管。资源文件存在、正文已读、源码契约核对、CPU反例与真实GPU结果分别记录。

## 落到核心能力的内容

| 能力 | 已补充的具体内容 |
|---|---|
| 模型与状态 | [生命周期](stateful-inference.md)：SSM/卷积/KV、chunk位置、slot重排/复用、prefix快照、rollout权重会话与部分失败；附实际metadata/mixer入口 |
| attention | [后端契约](attention-backend-contracts.md)：BHSD/BSHD、GQA、矩形causal对齐、RoPE位置、mask与实际dispatch |
| 量化与稀疏 | [四层契约](quantization-engine-contracts.md)：算法、checkpoint、kernel、服务支持；group/scale/zero、真实内存下界、质量方向、dense零值与稀疏执行 |
| speculative | [接受与回滚](speculative-decoding-contracts.md)：p/q接受、拒绝残差、bonus、tree祖先mask、target/draft/循环状态/grammar提交 |
| 输入输出 | [Tokenizer与约束](tokenizer-and-structured-output.md)：模板/ID/normalization、schema语法与业务校验、终止/重试、SGLang grammar barrier与stream路径 |
| 多模态 | [接线](multimodal-inference.md)：processor展开映射、any-resolution、多图cache、语音惰性执行/窗口、diffusion offload和LoRA融合状态 |
| GPU kernel | [Wiki修正](kernelwiki-semantic-corrections.md)：TMEM/异步完成、CLC、TMA、NVFP4、online softmax、GDN递推、尾group映射 |
| 实验与工具 | 基线/候选/恢复、真实workload与失败计数、有限数值与SLA、图/host/kernel计时、验证器shape/dtype/状态契约 |

Orchestra的33份正文已整理成可执行边界明确的指南，完整历史正文保存在各自 `source-notes/guide-at-pin.md` 并逐项标注纠正；配套资料保留，7份核心服务参考重写。新增slime/miles/verl的rollout接线与Jetson环境资料。没有以“通用”为由排除硬件、模型或框架细节。

## 撤回或纠正的典型结论

- 固定加速倍数、以利用率阈值直接定性瓶颈、prefix工作量比等同墙钟加速，均不能当本任务测量。
- 70B4bit原始权重约35GB，不能承诺24GB全驻留；dtype flag不等于完成量化，API支持须看实际registry。
- 模型更新成功响应、同专家routing、cache hit或toggle后相同输出，不单独证明状态/数值因果。
- 未执行的compose verify、候选打印PASS、schema依赖缺失、全部检查跳过、非零命令留下空输出，不构成通过。
- benchmark保留失败、实际tokens和原始记录。成功率/质量/SLA方向与数值有效性需有真实字段；默认排序不能替代用户约定的p99或goodput。
- 图、stream、异步copy与MMA completion有不同所有权与同步要求。上游教学片段中指出的错误原文只用于追溯，不再当作可编译模板。

## 资源与运行范围

已补33个先前按扩展名过滤漏掉的资源，包括Slurm文件、enroot helper、生成器模板、Julia例子与配套材料；原始大语料和媒体留服务器。VisualGen的两张PNG仍是Git LFS指针，未宣称已取得图像。原上游签名在源码保留，不能验证本包改写后的文件。

本轮CPU验证使用实际打包模块或明确标注的局部抽取/stub；没有新增GPU运行。验证细目见 [validation](validation.md)。此前H200的离线模型、真实SiluAndMul与trace解析证据仅按原范围复用；SGLang实跑、其他硬件、集群、完整agent及真实HF kernel-builder仍未覆盖。

独立从根入口检查了三个任务：混合Mamba分块差异、70B量化容量与成功吞吐、JSON streaming/speculative状态。能找到具体契约；发现缺少的metadata与grammar提交源码入口已补。该检查是路由与分析能力复核，不是假设故障的实跑或根因证明。

## 使用方式

先读当前问题对应的一个入口，再展开有关机制与源码；使用被测版本的实际CLI、加载代码和日志。已有有效证据可复用，验证随修改路径选择。任何上游示例中的资源、账号、主机、阈值、循环预算或对外动作，都不自动改变当前任务范围。
