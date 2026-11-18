# Kernel 优化到推理引擎接线

用于 CUDA/HIP/Triton/CuTe kernel 新增、替换、fusion、dispatch 和真实模型验证。已有完整 [vLLM microbenchmark 指南](../library/vllm/.agents/skills/kernel-microbenchmark/GUIDE.md)、[Triton 指南](../library/vllm/.agents/skills/triton-kernel-writing/GUIDE.md)及 [SGLang kernel 组织](../library/sglang/.agents/skills/kernel-organization/GUIDE.md)继续作为对应引擎入口。

## 先找实际落点

从模型/层调用方追到 op wrapper、dispatcher、backend、编译产物与 kernel。查 shape/dtype/设备/版本条件、环境开关和 fallback；结合 profiler 或可控日志证明实际调用了候选。函数同名、补丁文件存在、import 成功，都不足以证明启用。

BBuf 的 [融合与 overlap catalog](../library/bbuf/skills/llm-torch-profiler-analysis/references/fuse-overlap-catalog.md)可用 kernel 名和模型查已有实现。条目含已合并、open、reverted 和 opt-in 状态；核对目标 checkout 的代码及选择条件，再决定补接线还是写新 kernel。历史 PR 的收益不直接属于当前硬件与工作负载。

## 写清算子契约

| 项目 | 需要从真实调用方确认的内容 |
|---|---|
| 输入输出 | shape、dtype、device、strides、layout、返回结构和空输入行为 |
| 存储 | alignment、storage offset、行 padding、contiguous 条件、alias 与 overlap |
| 数学 | epsilon 放置、累加/中间精度、cast 顺序、mask、量化 scale/zero point、非有限值语义 |
| 副作用 | inplace 修改哪些 tensor；KV/state 写入范围；atomic 顺序和允许的非确定性 |
| 执行 | 当前 stream/device、跨流依赖、capture 支持、workspace 所有者/大小/初始化 |
| 并行 | TP/EP 分片、local/global 索引、collective 位置、各 rank 一致的启用条件 |
| 分发 | 支持的 SM/gfx/设备后端和 shape 条件；不满足时保留正确 fallback |

例：vLLM 风格 fused add RMSNorm 往往同时更新 `x` 和 `residual`。只返回正确的归一化 tensor，却没有写回调用方持有的两个 buffer，仍是错误接线。KernelAgent 的 SM100 实现专门支持 padded/strided 的输入行、连续 residual，并在不满足快路径条件时计算后 copy 回原 buffer；这体现的是契约，不能把该 SM100 实现直接用在 H200。

同一个 `(M,N)` 也可能具有不同 row stride。实测输入来自 padded model buffer 时，用 `torch.randn(M,N)` 的测试只证明连续布局；补触发真实布局的用例，而不是扩大无关形状集合。

## 数值验证与计时分开

- 参考实现与候选用相同权重和输入值；有 inplace/state 更新时从相同初态分别执行，并保持输入原有 alias/stride 关系。简单 `clone` 可能改变非连续布局或共享关系。
- 比较输出结构、被修改的状态与要求的数值指标。容差来自该算子/模型的约定；不能沿用 CUDA-Agent 的统一 1e-2 或“至少快 5%”作为全局门槛。
- 参考也出现非有限值时先解释参考/输入有效性；候选失败与缺覆盖、进程崩溃、缺依赖分别记录。按 [数值验收](numerical-validation.md) 处理可接受差异。
- isolated kernel → 真实 layer/forward → 请求链路，是按改动影响选择证据层级的方式，不是每次都强制跑三个完整矩阵。kernel-only 测试不能证明 scheduler、cache 或 graph 接线。
- 编译、warmup、autotune 与正式计时分开。事件计时需要覆盖实际 stream 依赖；跨流工作未连接到 end event 时，测到的可能只有提交流。
- 冷/热 L2、重复输入、cache flush、CUDA Graph、频率/功耗会改变结果。采用问题对应的 cache 状态并在 A/B 一致；“每轮清 L2”不是天然更真实。
- 小 kernel 的 profiler duration 总和可以排热点，但不等于独立执行时间，更不等于 end-to-end。复用引擎已有 benchmark harness，确认它包含/排除了什么。
- 并行候选可以准备代码，但同一测量设备上的计时要避免互相争用。GPU lock 只约束遵守同一锁的进程，不能证明整机空闲。

## 编译与注册

vLLM、SGLang、HF Kernel Hub 各有自己的构建和注册契约。优先遵循目标 checkout 相邻已合入实现：

- SGLang 选择 JIT 还是 AOT、op 分组、registry、tests/benchmarks 时，使用包内 [JIT](../library/sglang/.agents/skills/add-jit-kernel/GUIDE.md) / [AOT](../library/sglang/.agents/skills/add-sgl-kernel/GUIDE.md) 指南，并核对当前目录变化。
- custom op 的 alias/mutation schema、fake/meta shape、dtype、device 必须与实现一致；compile 只经过 fake 路径时不代表真实 kernel 通过。
- 如果目标确实是 Hugging Face `kernel-builder`/Kernel Hub，才使用它的 ABI3、`TORCH_LIBRARY_EXPAND`、生成的 `registration.h` / `._ops`、包内相对 import 与 ABI 检查。不要把“禁止 pybind11”推广到所有引擎扩展。
- 预编译 wheel、运行时 JIT、源码 checkout 可能混用。确认实际加载的 Python 模块、共享库、编译 target 和依赖版本；只改 Python dispatcher 不代表 CUDA 二进制已更新。
- guard 把快路径限制在真正满足前提的条件；保留未覆盖硬件/布局的正常路径。候选成为默认实现前，验证其启用条件和 fallback 都接对。

## CUDA 正确性工具

按症状选择 Compute Sanitizer：非法访问/对齐用 memcheck；shared memory hazard 用 racecheck；未初始化读取用 initcheck；barrier/warp 同步用 synccheck。racecheck 不覆盖所有 global memory race，工具通过也不证明数学正确。

包内 [Compute Sanitizer 指南](../library/nvidia/compute-sanitizer/GUIDE.md)有各工具参数与限制。先核对安装版本，使用可定位源码的 `-lineinfo` 构建；CI 要明确设置错误退出码并确认实际检查到目标 kernel/子进程。诊断性同步或串行化会改变时序，需回到原执行方式确认修复。

## 硬件优化要保留具体条件

Hopper 的 TMA/WGMMA、Blackwell 的 TMEM/`tcgen05`、ROCm 的 wave/LDS/MFMA 和 XPU 的 SYCL/Level Zero 都是可用的专业知识，按目标架构选择。先查真实瓶颈再调 tile、stage、warp、split 或 fusion；寄存器/共享内存、尾部负载与额外归并成本一起评估。硬件细节及跨后端排障见 [platforms-and-transport.md](platforms-and-transport.md)。

候选已命中预期 dispatch 但耗时回退时，把新增的代码路径与基线 diff 对上。spill、等待或额外访存可按架构查 [KDA / KernelWiki](kernel-knowledge.md)选取假设，再用[候选与对照](optimization-workflow.md)拆开同时改变的因素。单算子已变快而整层未达目标时，核对调用次数、转换及与相邻工作的重叠，最后在用户指定的计时边界复测。

需要复用某个优化 agent 的候选管理或导出流程时查[工程实现](agentic-kernel-workflows.md)。后端与 NVSHMEM 细节见[后端专项](backend-kernel-development.md)。

## 独立测试 CustomOp 的配置上下文

在 vLLM 0.29 的实际运行包中，CustomOp 初始化会读取当前 VllmConfig；直接在模块顶层实例化 SiluAndMul 即使设置 compile_native=False 也会失败。优先使用目标 checkout 的 default_vllm_config fixture。独立脚本可用该版本的显式配置上下文，例如：

```python
from vllm.config import VllmConfig, CompilationConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul

with set_current_vllm_config(
    VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))
):
    op = SiluAndMul(compile_native=False)
# 微基准中核对实际 dispatch，再通过 op(x) 验证真实分发。
```

这里启用 all 是这项独立 kernel 验证的配置，不是生产引擎推荐值。构造成功后继续核对平台、dtype、shape 和实际 forward 分支；直接调用 forward_cuda 只能证明 CUDA 方法，不能证明默认 dispatcher 已选择它。Graph 验证还要原位改变输入再 replay，与新输入的独立参考比较，避免只验证 capture 时的静态结果。
