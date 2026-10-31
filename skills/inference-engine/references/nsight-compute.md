# NCU：从真实热点到 kernel 内部原因

当已有 Torch/Nsys 证据指向具体 kernel，或正在实现算子时读取。NCU 用于解释单个 launch 的资源和指令行为；其 replay 计时不能替代无 profiler 的引擎性能对照。

## 先绑定被测对象

记录 kernel 的完整名称、实际 dispatch、shape、dtype、stride/layout、batch、KV 长度、rank、GPU 架构和二进制来源。挑中真实热点的代表性 launch；不要把初始化、JIT、autotune 或只在 fallback 出现的同名 kernel 当成目标。

独立 harness 保留真实输入布局、mask、量化 scale、分片、输出/残差别名和必要的依赖。随机连续 tensor 只能覆盖相应条件。对融合或状态 kernel，显式恢复输入和状态；连续 replay 可能改变 inplace 数据或 KV/state 池。

已有报告的导出和 Python 解析是 CPU 工作。新采集须在当前任务 GPU 授权内；不要因导入报告而启动推理服务。

## 采集和版本选择

先查看目标安装的 `ncu --version`、`--help`、`--list-sections`、`--query-metrics`。下面是候选命令形状，section、filter、skip/count 按该版本与实际 launch 确认：

```bash
ncu --target-processes all --kernel-name-base demangled \
    --kernel-name 'regex:TARGET_KERNEL' --launch-skip SKIP --launch-count COUNT \
    --section SpeedOfLight --section LaunchStats --section Occupancy \
    --section MemoryWorkloadAnalysis --section WarpStateStats \
    --export "$RUN_DIR/target" python reproduce.py
```

先采回答当前问题所需的 sections；source/PC stall 或 PM sampling 确有需要时再加。全量 `--set full` 会增加 replay 次数，不作为每个 kernel 的默认要求。

- 保留与正式优化相同的构建优化，添加 `-lineinfo` 便于定位；`-G` 会改变生成代码，不能用于正式性能结论。
- 记录 replay 模式、cache control、clock control、PM sampling 与并发条件。kernel replay、application replay、range replay 的适用范围不同；不能只改模式而忽略重复运行的服务状态、外部通信和输入一致性。
- 多 rank collective 或跨 kernel 同步可能在隔离/replay 后改变行为甚至挂住。先确认该模式支持目标执行；需要端到端时间关系时回到 Nsys。
- 报告为空先查 process/launch filter、采集窗口、CUPTI 注入冲突和权限日志。工具之间的兼容问题以实际错误为证据，不自动升级镜像或修改系统计数器权限。
- 不从工具发现失败推断模型错误，也不从文件存在推断采集成功。

## 解读顺序

| 信号组合 | 可能原因 | 下一步证据与动作 |
|---|---|---|
| grid 小、waves 少、SM 活跃不均 | 并行工作不足或尾部不均 | 核对每 CTA 工作量、KV 长度分布；评估 split-K/分段归约或分派，但计入归并成本 |
| DRAM 流量与吞吐接近实测可达上限 | 带宽约束 | 算实际 bytes，核对重复读写/量化/融合机会；别用厂商峰值替代可达上限 |
| long scoreboard 高、DRAM 吞吐低 | 数据依赖、访存延迟或不足的并行访存 | 定位 load 的 PC、coalescing、cache 与独立工作；不能只称“带宽瓶颈” |
| shared 实际 wavefronts 超过理想值 | bank conflict 候选 | 对具体指令和布局核对；广播、低访问量与规则估算误差要区分 |
| register 数、local load/store 增加 | spill 或局部数组访问 | 检查编译器报告/SASS、生命周期和 tile/stage；压寄存器可能进一步增加 spill |
| barrier/short scoreboard 集中 | 同步、共享内存依赖或工作不均 | 看 producer/consumer 与 barrier 的必要性；不能直接删同步 |
| FP64 或非预期指令路径 | 精度提升、类型/常量或 fallback | 对照源码和 SASS；改变数学函数或累加精度需回到数值验收 |
| occupancy 低 | 小 grid、资源上限、短任务或尾部等 | 同看理论 occupancy、寄存器/共享内存和实际时长；提高 occupancy 不是独立目标 |

Roofline 以目标 dtype、实际执行的计算和存储路径为准；区分 requested bytes 与 DRAM bytes、dense 与 sparse 峰值、L2 与 HBM 带宽。PC sampling 的 stall 份额不是墙钟延迟占比，源码行被多个 inline frame 共享时不能机械累加。NCU 的 estimated speedup 是分析线索，不是可相加的收益承诺。

H100/H200 属于 Hopper 路径，Blackwell 专有 TMEM/`tcgen05` 示例不直接用于它们。即使算力架构相同，SKU 的 SM 数、缓存、带宽、功耗和有效 shape 也须实测/查询；不把 H100 的可达值套给 H200。

## 已有报告的离线分析

包内保存了 MIT HAN Lab 的完整 [NCU 指南](../library/ncu-report/GUIDE.md)、[分析维度](../library/ncu-report/reference/05-analysis-dimensions.md)、[诊断手册](../library/ncu-report/reference/06-diagnosis-playbook.md)、[指标表](../library/ncu-report/reference/08-metric-names.md)和配套 helpers。该快照主要围绕 NCU 2026.3 / Blackwell；具体指标是否存在以报告和工具为准。

```bash
ncu --import "$REPORT" --page details
ncu --import "$REPORT" --page raw --csv
python3 "$SKILL_DIR/library/ncu-report/helpers/ncu_utils.py"
python3 "$SKILL_DIR/library/ncu-report/helpers/analyze_reports.py" \
    --run-dir "$RUN_DIR" --report "$BASE_REPORT" --tag baseline \
    --report "$CANDIDATE_REPORT" --tag candidate --kernel "$KERNEL_REGEX"
python3 "$SKILL_DIR/library/ncu-report/helpers/extract_stall_hotspots.py" \
    --run-dir "$RUN_DIR" --report "$REPORT" --tag candidate --kernel "$KERNEL_REGEX"
```

Python helpers 依赖 Nsight Compute 附带的 `ncu_report`（必要时把对应安装的 `extras/python` 加入 `PYTHONPATH`），不依赖 GPU 执行。当前包仅做过源码/语法检查，尚未用真实 NCU 报告验证这些新增 helpers；依赖缺失时用可用的 CLI 导出或保留待解析状态。

检查输出包含所有请求的 tag/action。上游 helper 遇到报告不存在或 kernel 不匹配会打印 `[skip]` 并继续，退出码 0 不能证明对照齐全。缺 metric 保留 null/unknown；明确是未采集、不支持还是读取错误。源行未知也保留在统计中。最后回到无 profiler 的同条件测试确认修改收益。
