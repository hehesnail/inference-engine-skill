# KernelWiki 技术契约与逐页修正

这是对锁定 KernelWiki 48 篇技术正文的语义审阅补充。来源标签、schema 合法、代码 provenance、编译成功、数值通过、性能有效是不同证据。原页的 `snippet` 只表示存在片段，不表示本次编译或执行；本轮没有运行 GPU，也没有逐条重验 2179 条 PR 的历史事实。逐页修正置于每页正文之前，后面的上游叙述保留供溯源，其中被指出的代码不得作为实现模板。

## SM100 的异步资源所有权

`tcgen05.mma` 的单线程发起和 TMEM 分配/读写的 warp 集体要求不同。分配由一个 warp 的所有 lane 以一致参数执行；地址写入 shared memory。SM100 合法分配列数为 32/64/128/256/512，每个 SM 有自己的 128×512×32-bit TMEM；两 CTA 模式不把两个 SM 简化为一个银行的前后256列。TMEM ld/st 需服从指令形状和 warp 位置的行映射；不能让任意 epilogue warp 用普通二维索引读取任意行。`tcgen05.cp` 的方向是 SMEM→TMEM。[PTX 8.7 分配与访问说明](https://docs.nvidia.com/cuda/archive/12.8.0/pdf/ptx_isa_8.7.pdf)、[CuTe 原语和布局](https://docs.nvidia.com/cutlass/4.8.0/media/docs/pythonDSL/primitives.html)。

MMA issue、线程同步、proxy fence、异步完成是四件事。producer 等 free-slot，注册本轮 TMA 总字节并发起拷贝；consumer 等 full-slot 后使用匹配的 visibility fence/descriptor 发 MMA；在 MMA 真正完成后，才能释放其 SMEM 输入或向 epilogue 发布 TMEM 结果。使用 `tcgen05.commit...mbarrier...` 的完成通知与等待，不用普通 `mbarrier.arrive` 或 `__syncwarp` 冒充 MMA completion。TMEM ld/st 还有对应 wait；跨线程接力需正确 fence。下面只表达生命周期，不是 CUDA API：

```text
for each logical tile:
    producer waits for this slot's prior consumer completion, if reused
    producer arrives with the total expected TMA bytes; issues all copies
    consumer waits for this slot's full phase
    consumer establishes required visibility and issues MMA
    commit completion notification for the issued MMA
    after completion, publish/free only the resources that operation finished using
```

每个循环 slot 有自己的 phase；mbarrier 仅在到达计数与未完成事务计数都满足时完成一个 phase，不是每条 arrive 都翻转。barrier 计数要匹配实际参与者，不能 init(1) 却让14个warp分别 arrive。mbarrier 本身从 sm80 起，TMA 等组合的架构要求另查。[PTX 11.5.1](https://docs.nvidia.com/cuda/archive/11.5.1/parallel-thread-execution/index.html)、[tcgen05 completion 原语](https://nvidia.github.io/cccl/unstable/libcudacxx/ptx/instructions/tcgen05_commit.html)。

TMA 的 datatype、rank、box、stride、alignment 和 swizzle 是描述符契约，不是任意 dtype 转换器。global 地址的基本16-byte约束与某些格式的更强约束、SMEM swizzle 的对齐不同，不能推出所有 NVFP4 的 K 必须256/2048。none/32/64/128-byte swizzle 都有各自合法配置；128B不是所有 MMA 的唯一正确布局。[CUDA tensor-map API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__TENSOR__MEMORY.html)。

16 warp CTA、14 epilogue warp、3至5 pipeline stage 是特定实现的选择。tile/layout、寄存器、SMEM、TMEM、CTA resident 数及短 K 开销共同限制调优空间。双buffer与warp specialization可用于Hopper；不能把一次教程的组合收益归因单个机制。

## CLC 与 tile 映射

CLC 请求取消一个**尚未启动**的 CTA/cluster，成功时调用方接手它的工作；不是取消调用者自身。先计算当前 blockIdx 对应工作，再发异步取消请求、等待完成、检查成功，成功才读取被取消坐标。观察到失败后不再提交新的取消请求。grid 必须覆盖逻辑工作集合；只发 SM 数 CTA 再凭空查询全部 tiles 是不同的软件队列设计，不能冒充 CLC。[CUDA CLC 官方流程](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-programming-guide/04-special-topics/cluster-launch-control.html)、[CUTLASS CLC 工作映射](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_cluster_launch_control.md)。

CLC 不含页内虚构的 `clusterctl.init/query/wait` 或 `Hilbert` 硬件策略接口。tile swizzle 是软件坐标映射。工作只有8个不可分tile时，调度器不能把它变为142份并行计算。split-K或更小tile改变粒度，也引入归约/访存/同步成本。

下面按列分组的 CPU 映射覆盖任意正整数尺寸和最后不满组，保留 A/B reuse 顺序的调优空间；它不是硬件 CLC API：

```python
def grouped_raster(index, tiles_m, tiles_n, group_cols):
    if min(tiles_m, tiles_n, group_cols) <= 0:
        raise ValueError("positive tile counts and group size required")
    if not 0 <= index < tiles_m * tiles_n:
        raise IndexError(index)
    width = min(group_cols, tiles_n)
    group = index // (tiles_m * width)
    first_n = group * width
    actual_width = min(width, tiles_n - first_n)
    offset = index - group * tiles_m * width
    return offset // actual_width, first_n + offset % actual_width
```

## NVFP4、FP8 与错误预算

NVFP4 使用 E2M1 数据、每16元素的 E4M3 block scale，并按格式契约应用 global scale；MXFP4 常用 UE8M0 与不同 block 大小。两者不可只改dtype名或把 E4M3 scale量化为UE8M0而声称同一计算。SM100 block-scaled 指令中，`mxf4nvf4` 支持 E2M1 + UE4M3 的组合；scale 的物理布局/驻留、instruction descriptor 与 pack顺序必须匹配真实原子。`kind::f8f6f4` 不等同上述 native NVFP4 路径。[PTX block-scale 类型表](https://docs.nvidia.com/cuda/pdf/ptx_isa_9.0.pdf)。

独立 CPU 参考先完整解码所有nibble与scale，再验证 GEMV/GEMM：uint8数值转换不是E4M3解码，加载64bit只转换其中两个nibble会漏元素。scales、global_scale、K尾、padding、batch/row stride都属于ABI。两个256列 accumulator 已用满SM100 TMEM，不能忽略 native scale等其他分配需求。Hopper FP8 promotion interval 的128通常指 K 元素间隔，不是128次WGMMA；依据实际固定版本和循环步长解释。

## Attention 的归一化不变量

设本block有效score为 `s`，value为 `v`，旧状态为 `(m,l,a)`，其中 `a` 是未归一化 weighted-value sum。对一个非空有效block：

```text
m_new = max(m, max(s))
alpha = 0 if l == 0 else exp(m - m_new)
p = exp(s - m_new)              # invalid/masked entries contribute zero
l_new = alpha*l + sum(p)
a_new = alpha*a + sum(p[:,None]*v, axis=0)
output = a_new/l_new            # only after checking l_new>0
```

不能用 `logaddexp(old_lse, block_max)` 代替 block logsumexp；两个零score、两个value=1时正确输出是1，旧 NSA 页公式给2。不能更新running max却不相应重标旧accumulator与denominator；如做近似延迟 rescale，必须维持另一套明确的归一化参考并测量误差。全mask/空selected-index输出和NaN语义按引擎契约处理。

NSA 的块选择/压缩/滑窗组合与 DeepSeek V3.2 DSA 不是同一个算法。DSA 的选择单位及输出index布局要与真实FlashMLA消费端一致；top-k主attention的复杂度不包含选择阶段成本。656byte KV cache属于特定MLA数据格式，不是所有模型或indexer的通用条目大小。

## Gated Delta Net 的状态契约

使用 row-vector 约定：`S` 为 `[K,V]`，`q,k` 为 `[K]`，`v` 为 `[V]`，`beta` 为写入强度，`g_log` 为log-space decay。一次序列步的参考为：

```python
S_decay = exp(g_log) * S
residual = beta * (v - k @ S_decay)
S_new = S_decay + outer(k, residual)
o = (q * query_scale) @ S_new
```

若模型传raw gate/logit、已sigmoid的beta、归一化q/k或多value heads，adapter必须明确转换/广播，不能靠变量名猜测。与 [FLA recurrent reference](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/naive.py) 对应。chunk实现需要块内因果关系和跨chunk state传递；多个CTA直接读写同一State不是等价并行化。比较初始state、每步output和final_state，并测试非零/弱decay状态；不能只测强衰减使错误被掩盖的样本。

## exp、多版本与测量归因

旧 software-exp 页给的是四阶 Taylor 系数，不是经过证明的minimax。实数计算在 x=-0.5 时相对误差约5.568e-5，约14.13bit而不是22bit；浮点实现还需额外误差分析。不能从128个FMA单元/16个SFU推出8倍exp吞吐，4个相依FMA、range reduction和reconstruction都有成本。展示的PTX重建仍调用ex2，更不能称完全绕过SFU。候选需用实际生成指令验证，保存误差域、极端值与端到端收益。

Triton3.6前没有TMEM/tcgen05的说法错误：[3.5.0 release](https://github.com/triton-lang/triton/releases/tag/v3.5.0)已记录TMEM与Blackwell相关代码生成。是否使用tcgen05由实际版本、dtype、shape、API和target共同决定，须检查生成代码。GB300是CC10.3，不能作为精确sm100同义词；检索可以使用Blackwell family alias，但编译目标需实际检测。[NVIDIA compute capability](https://developer.nvidia.com/cuda/gpus)。

256-bit `.v4.b64` 支持不能推广为sm90通用能力；PTX向量加载需要合法对齐，不能假定误对齐自动退化安全窄load。向量化主要减少指令开销，32bit但充分合并的访问并非必然浪费bus；cache hint不是驻留保证；inline PTX仍由后端寄存器分配。[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)。

NCU 的SM/Memory SOL不是整图端到端速度，也不自动等于DRAM带宽利用率。按launch身份、活动窗口和metric定义做归因，保留所有相关kernel及copy；不要只取首个非`at::`名字。框架wall、CUDA events、kernel duration sum、profile重放和prepare/run分别报告。所有历史吞吐值保留为来源陈述，不作为当前模型的接受阈值。[Nsight Compute triage](https://docs.nvidia.com/nsight-compute/ComputeTriage/)。
