# 已安装的离线工具

`scripts/offline_tools.py` 默认使用同一 skill 目录下的 `library/`。按工作区配置选择计算环境，再将 `SKILL_ROOT` 设为该环境的安装目录：

```bash
SKILL_ROOT=/path/to/inference-engine
python3 "$SKILL_ROOT/scripts/offline_tools.py" --list
python3 "$SKILL_ROOT/scripts/offline_tools.py" profile --help
```

工具使用 Python 标准库；`plan` 另需 PyYAML。已有 trace/log 分析无需安装推理框架、PyTorch 或 CUDA。相对输入路径以调用者当前目录为准，建议传该环境的绝对路径。每个工具可用 `<工具名> --help` 查看完整参数，不会触发推理负载。以上为 shell 示例，使用其他 shell 时按其变量语法传入同样的参数。

资料与入口不在一起时，在工具名之前传 `--library-root /path/to/library`，或设置 `INFERENCE_SKILL_LIBRARY`；该目录应直接包含 `bbuf/`。显式参数优先于环境变量。`--list` 标明工具脚本 `available` / `not-installed`；这只检查脚本存在，历史语料、导入依赖和配置仍需齐备。迁移某项工具时连同其依赖与参考一起复制，不能只复制一个脚本。工作区要求远程计算时仍在远程执行，工具本身不限制操作系统。

| 工具名 | 作用 | 完整说明 |
|---|---|---|
| `profile` | 单 trace 或 eager/graph 双 trace 的 kernel、overlap、fusion 分析 | [profiler](../library/bbuf/skills/llm-torch-profiler-analysis/GUIDE.md) |
| `capacity` | 启动日志、KV 与显存分解 | [capacity](../library/bbuf/skills/llm-serving-capacity-planner/GUIDE.md) |
| `compare` / `plan` | 比较归一化结果；校验 cookbook 并生成候选命令，不启动服务 | [benchmark](../library/bbuf/skills/llm-serving-auto-benchmark/GUIDE.md) |
| `pipeline` / `layer-kernels` / `time-map` | forward/layer 边界、逐层 kernel 与 Perfetto 时间定位 | [pipeline](../library/bbuf/skills/llm-pipeline-analysis/GUIDE.md) |
| `compute` / `compute-trace` | config 推导理论 shape/FLOPs；带映射假设的 trace 对照，不能直接作实测 MFU | [compute](../library/bbuf/skills/model-compute-simulation/GUIDE.md) |
| `layer-track` / `compact` | 添加有依据的层标记、压缩展示轨道；保留原 trace | [layer track](../library/bbuf/skills/torch-profiler-layer-track/GUIDE.md) |
| `review-query` / `review-summary` | 按问题查历史讨论，或做有统计覆盖的完整 sweep | [review](../library/bbuf/skills/sglang-humanize-review/GUIDE.md) |
| `model-history` | 按框架/模型查 PR 历史档案 | [history](../library/bbuf/model-pr-optimization-history/GUIDE.md) |
| `day0-check` | 检查模型支持资料包的字段和证据结构 | [Day-0](../library/bbuf/skills/model-optimization/sglang-model-day0-support/GUIDE.md) |

常见调用（将示例路径替换为本次文件）：

```bash
python3 "$SKILL_ROOT/scripts/offline_tools.py" profile \
  --framework vllm --input "$RUN_DIR/worker.trace.json.gz"
python3 "$SKILL_ROOT/scripts/offline_tools.py" capacity \
  --log-file "$RUN_DIR/server.log" --format json
python3 "$SKILL_ROOT/scripts/offline_tools.py" review-query \
  --query 'cuda graph' --limit 5
python3 "$SKILL_ROOT/scripts/offline_tools.py" model-history \
  --framework vllm 'qwen3 fused qk norm'
```

离线入口拒绝 URL/host/port 参数，子进程不暴露 GPU。它不会安装软件、下载数据或自动改成在线采集。完整参考中还保留服务启动、HTTP profiler、请求重放、GitHub 收集和 kernel GPU 实验等脚本；它们不在离线入口内，使用时核对当前环境与授权。

本次验证范围见 [validation.md](validation.md)。资料位置和恢复方法见 [server-data.md](server-data.md)。

## 可选 NCU 报告 helpers（依赖单独说明）

新保存的 [NCU helpers](../library/ncu-report/helpers/README.md)可对已有报告抽取 metrics、PC/source stall 和 PM 时间线。调用、版本与输出完整性检查见 [nsight-compute.md](nsight-compute.md)。它们不通过 offline_tools.py 暴露；需要 Nsight Compute 自带 ncu_report，目前维护环境缺依赖，尚未验证真实报告运行。不得将语法检查或退出 0 当成完整分析通过。

TRT Host 脚本在 [专题资料](../library/trtllm/host/GUIDE.md)中保留作实现参考，不能未经适配直接用于 vLLM/SGLang trace。原稳定离线工具及可用性判断不变。

## 新增知识检索与导出

使用 scripts/research_tools.py --list 查看脚本及语料是否齐全；find按关键词和--repo查询302个完整专项，wiki-query/wiki-page/wiki-grep使用KDA锁定知识库。参数与来源边界见[KernelWiki](kernel-knowledge.md)。kda-export调用随包原导出脚本，需要Python3.11+及kernel-builder；导出不包含发布。

## 语义核对后的工具边界

自动排名会核对已有 success/SLA 和记录的 workload/resource 一致性，但不能证明原始数据、质量或cache初态真实一致。`compute` 的多算子、多precision映射是估算模型，已有漏配反例；实际per-kernel MFU必须核对具体运算与dispatch。KernelAgent候选输出中的PASS字符串及未执行的compose verify也不能代替独立输出/state验证。具体修正与仍需现场证据的项见 [语义审阅](semantic-audit.md)。
