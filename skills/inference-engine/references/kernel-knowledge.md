# KDA / KernelWiki：从症状找到代码与证据

使用 KDA 锁定的 KernelWiki 提交 76d27b56f804e7e7295d4c570e1e5d7eef4b0a75，不自动切到 upstream main。内容截止 2026-04-27：48 篇技术页、2179 条 PR、20 篇博客、11 份文档、7 份竞赛材料，以及代码/patch/PROVENANCE。截止日不是本次安装日期。

全量知识库与两个KDA子模块保存在服务器完整包；桌面保留技术页、数据schema、查询脚本和入口。完整PR与代码语料按工作区存储策略保留服务器；缺语料时工具会报“未安装完整语料”，不会返回一个伪装成全库查询的空结果。

## 实际调用

在配置允许的计算环境运行。SKILL_ROOT 是当前 skill 路径；LIBRARY 是包含完整 kernelwiki 的 library 目录，可用 INFERENCE_SKILL_LIBRARY 设定。

```bash
python "$SKILL_ROOT/scripts/research_tools.py" --library-root "$LIBRARY" wiki-query -- "UMMA" --architecture B200 --limit 5 --compact
python "$SKILL_ROOT/scripts/research_tools.py" --library-root "$LIBRARY" wiki-query -- --symptom pipeline-stalls --compact
python "$SKILL_ROOT/scripts/research_tools.py" --library-root "$LIBRARY" wiki-page -- kernel-flash-attention-4 --follow-sources
python "$SKILL_ROOT/scripts/research_tools.py" --library-root "$LIBRARY" wiki-page -- pr-cutlass-2472 --include-code
```

先query看id/path/architecture/confidence，再page读正文，随后沿sources取完整原文与代码。follow-sources只给500字符摘录，不等于完整读取；include-code可能截断单文件，做修改前读实际文件。工具需要PyYAML，不需要GPU；--list区分脚本存在与完整语料可用。

confidence的verified/source-reported/inferred/experimental和reproducibility的snippet/runnable/benchmarked是上游资料分类。读取PROVENANCE中的commit与具体shape/dtype/硬件后，才决定是否适用于当前引擎。历史性能数字不当验收门槛；没有新commit核对就不称当前release已包含某优化。

## 选题路径

| 问题 | 技术页及接线 |
|---|---|
| Hopper→Blackwell GEMM | WGMMA→tcgen05、register→TMEM，两份迁移正文；再查TMA/fence/barrier |
| Tensor core低利用率、pipeline stall | pipeline stages、double buffering、ping-pong、warp specialization，核对每个stage生产/消费与phase |
| memory-bound/寄存器/尾部 | vectorized loads、cache policy、swizzle、register budgeting、persistent/CLC；不要仅凭occupancy归因 |
| FP8/NVFP4/MoE | scale布局/精度、grouped/gated dual GEMM、MoE load imbalance、mask与routing |
| attention/MLA/线性attention | FA4、FlashMLA、sparse MLA、NSA、Gated Delta Net；接回真实prefill/decode和state更新 |

完整48页入口见 [技术页目录](kernel-wiki-pages.md)。

48页已逐项语义审阅；[同步、数学、格式与编译契约修正](kernelwiki-semantic-corrections.md)与各页开头的具体纠正优先于保留的上游教学片段。逐页读取不等于所有GPU代码已运行，也不等于2179条PR的全部历史事实已重验。

## 已核实的修正

- **PDL/GDC：** 不是所有Blackwell kernel默认自动重叠。显式设置 launch 的 programmatic serialization属性；生产者用cudaTriggerProgrammaticLaunchCompletion，消费者在读取依赖结果前用cudaGridDependencySynchronize或等价同步。并发仅是机会，不是正确性依赖。对应页面已修正，原文另存以便追溯。[CUDA 官方说明](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)
- **mbarrier：** 多stage轮转要维护每个stage的phase，不能每轮只翻一个全局phase；arrival与未完成transaction共同决定完成。页内伪代码已改为逐stage phase，仍需匹配实际producer/consumer协议。
- **硬件容量：** SM100 shared memory每SM228KB、单block最多227KB；SM120的限制不同。两者不能笼统共用Blackwell数值。[NVIDIA架构指南](https://docs.nvidia.com/cuda/archive/13.0.0/blackwell-tuning-guide/index.html)
- 竞赛绕过harness/缓存输出的案例用于识别无效测量，不是候选优化方法。性能“上限”、固定占用率/流水线深度和经验收益保留为来源观察，不推广。

## 两套NCU与Atrex知识依赖

KDA锁定的NCU为d1887948c7d53690cfe6605f59c1329b8a1c6bb5；独立NCU包为c7980a43cb88a8734d113b22b57ff66746cf5719。两套原文保留，日常入口用[NCU](nsight-compute.md)，不混用不同版本helper/metric名称。

Atrex锁定另一版KernelWiki和GPU编程教材，完整依赖位于其服务器checkout。KDA的source ID和Atrex runtime查询协议不同；本地wiki查询不需要启动Atrex Gateway。更新时分别记录各自gitlink、license和内容摘要。
