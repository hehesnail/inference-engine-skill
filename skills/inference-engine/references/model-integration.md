# 自定义模型与状态接入引擎

用于新增模型、替换执行路径或引入额外状态。按当前改动沿实际数据流核对；这些是定位问题的线索，不是每次必须跑完的测试矩阵。

## 按需读取实现资料

vLLM 包内保存了 [模型接入](../library/vllm/docs/contributing/model/basic.md)、[模型测试](../library/vllm/docs/contributing/model/tests.md)、[Model Runner V2 设计](../library/vllm/docs/design/model_runner_v2.md)与 [model-microbenchmark](../library/vllm/.agents/skills/model-microbenchmark/GUIDE.md)。先确认目标版本是否采用相应 runner、接口和命令；旧 wheel 不一定有 `vllm bench forward`。SGLang 从 [sglang.md](sglang.md) 的模型、运行时和 kernel 指南进入，并以当前 checkout 为准。

## 沿完整生命周期定位

1. **加载与启动**：模型注册、config、权重名称/分片、dtype 与设备放置；dummy/profile run、显存估算和图捕获是否会在缺少正常请求元数据时调用模型。启动成功不证明正式请求、容量或评分正确。
2. **调度与缓存**：scheduler 是否接受该模型导出的 cache 规格；worker 如何分配、绑定、传递缓存与 attention metadata。若替换 runner，检查其与现有 scheduler 的接口，不能仅凭模型 forward 单测认为已经接入。
3. **请求与模型状态**：输入 token、positions、序列长度、slot/block 映射及额外循环状态分别由谁拥有。沿新增、续跑、chunk、结束、取消和复用等本次影响路径核对生命周期；独立请求不能意外共享残留状态。
4. **输出与评分**：runner 可能只把选定 hidden rows 交给 logits/head，也可能为 prompt logprobs 或评分再次调用。核对 shape、索引与调用次数；输出转换不应意外推进模型递归状态或重复写缓存。
5. **预算与优化分支**：除 KV 外，状态池、临时 workspace 和 CUDA Graph 也影响可用容量。记录哪些 flag、shape 或阶段进入新路径、何时回退，以及实际默认值。

混合 attention/循环模型需要读真实状态布局与所有权，不能只按普通 KV cache 推断。记录哪些对象被原地修改、哪些随请求重排或释放，以及设备与 stream 的依赖。

## 选择有效验证

若任务是为 SGLang 新模型准备首日支持，按 [Day-0 工作流](../library/bbuf/skills/model-optimization/sglang-model-day0-support/GUIDE.md)将架构差异、实现、验证矩阵和实际证据整理成材料包。`python "$SKILL_ROOT/scripts/offline_tools.py" day0-check BUNDLE --format json` 可以发现未填字段和材料问题。模型接入是否成功仍由对应版本的真实加载、推理输出及任务验收决定。

先找当前项目可复用的接入/端到端用例和已验证结果。覆盖本次修改经过的启动、调度、执行、输出接线，必要时用隔离 CPU 用例定位契约；CPU 模拟与真实 GPU 生成分别报告。数值判断遵循 [numerical-validation.md](numerical-validation.md)。

有状态微基准先确定重复调用是否从相同状态开始；reset 可能涉及 KV、循环状态、长度计数和临时池。只重置 KV 不一定构成对称对照，反复累积状态也可能测到不同工作量。性能计时与状态初始化的边界需与基线一致。

实现存在、分支启用和性能有效是三类证据。交付沿一个输入说明其经过的状态/接口与输出变化，列出有效条件和未覆盖路径；不将某次模型适配的固定阈值或服务器设置写成通用规则。

## 涉及特定状态时继续读取

- [循环状态与在线权重更新](stateful-inference.md)：SSM/卷积状态、prefix共享、slot复用、rollout版本与部分更新失败。
- [Attention布局、位置和mask](attention-backend-contracts.md)：BHSD/BSHD、矩形causal mask、GQA、RoPE和真实backend。
- [多模态接线](multimodal-inference.md)：processor、媒体token、encoder cache、语音窗口与diffusion状态。
