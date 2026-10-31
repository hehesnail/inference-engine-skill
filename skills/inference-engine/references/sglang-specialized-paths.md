# SGLang 新增路径与完整专项

原有 27 个 root 专项保留，本次同步上游改动并补入 multi-graph 和 5 个 diffusion 专项。完整目录见 [SGLang 目录](catalog.md) 与 [工程目录](repository-guides.md)。

## CUDA graph lifetime

读 [multi-graph 完整指南](../library/sglang/.agents/skills/sglang-multi-graph-check/GUIDE.md) 和随附 CLI。工具实现位于目标 checkout 的 sglang.test.cuda_graph_lifetime，不能把这个 skill 的导入路径当被测服务源码。

区分新运行、只读归档和归档 recompute。只读归档不需要GPU或启动服务；recompute可能改写结果，应使用本任务允许的副本。新运行沿已定配置追踪worker、model runner、attention/LoRA/MoE、draft/verify、communication workspace及独立graph，声明可检查、仅可观察、partial、unsupported、unknown。

分别报告 lifetime、numerical、coverage；每worker检查 capture、inspection和真实replay，capture成功不能代替实际bucket被运行。大小归一后的expect-label不能证明每个具体bucket。allocator外内存、device pointer tables、未更新但仍存活的metadata和未发送traffic属于其限制。recompute退出0也不等于这些verdict全通过，不扩大已约定测试矩阵。

## Diffusion模型接入

[add-model](../library/sglang/python/sglang/multimodal_gen/.agents/skills/sglang-diffusion-add-model/GUIDE.md) 保留三种pipeline：
- Hybrid：模型特定 BeforeDenoisingStage，标准 denoise/decode；
- Modular：复用现有 text/image/latent/timestep stage；
- Native task contract：耦合音视频/多请求形态，保持 canonical request 从 admission 到 offline/server 一致。

接线包括模型组件、config、sampling params、pipeline、registry、stage输出字段以及参考实现对照。先 eager parity，再选择 graph、compile、量化或缓存；不能仅因属于allowlist就声称性能路径已验证。

## 性能与缓存

[benchmark-profile](../library/sglang/python/sglang/multimodal_gen/.agents/skills/sglang-diffusion-benchmark-profile/GUIDE.md) 与 [performance](../library/sglang/python/sglang/multimodal_gen/.agents/skills/sglang-diffusion-performance/GUIDE.md) 区分 denoise、VAE、编码、传输和请求端到端。同步stage计时会改变执行行为，归因测量与最终性能分开。确认native backend，Diffusers fallback不能标成原生SGLang结果。

BCG需真实capture/replay及signature命中；分辨率之外，帧数、conditioning、packed host边界也影响fallback。SP/Ulysses/Ring/CFG、offload与prefetch按拓扑和各组件每请求调用次数选择。VAE按temporal chunk重复执行时，“仅执行一次的组件”假设失效；保留host RAM、GPU peak和传输成本。

[cache-dit](../library/sglang/python/sglang/multimodal_gen/.agents/skills/sglang-diffusion-cache-dit/GUIDE.md) 的 DBCache/DMD/TaylorSeer/SVDQuant、per-request knob和状态重置分别检查；固定seed/输入/分辨率/步数，PSNR等质量统计按任务约定与无缓存参考比较。参数存在不证明命中目标cache path。

[ModelOpt量化](../library/sglang/python/sglang/multimodal_gen/.agents/skills/sglang-diffusion-modelopt-quant/GUIDE.md) 的校准、导出、加载、scale/layout与backend选择作为一条链验证。FP8/NVFP4和高质量近似优化保留各自适用硬件与数值条件。

不继承上游默认关闭版本检查、每轮下载后清模型缓存、强制新建环境或默认全模型矩阵；这些仅在当前任务需要时使用。完整脚本保存供明确选定的benchmark使用，尚未运行GPU。

## processor parity更新

新的processor-model-parity覆盖request→normalize/validate→render→tokenize→/generate→reasoning/tool parsing→OpenAI response。文本一致、token IDs一致与真实host接线分别验收；tokenizer缓存缺失时上游测试可跳过token检查，不能仅凭cargo退出0报告token parity。固定checkpoint revision，测试fixture由Python参考路径产生，不能改expected来迁就Rust结果。

## 跨模型输入与状态

Vision/audio/diffusion的processor、窗口、adapter和计时边界见 [多模态推理](multimodal-inference.md)。混合SSM与训练rollout接线另读 [状态与权重更新](stateful-inference.md)，包括SGLang更新会话、target/draft部分失败与cache flush限制。
