> 本地核对：特定模型层数、固定CPU/GPU时间偏移和window筛选是来源样例。当前trace需核对clock/rank/device、窗口裁剪与未知事件，不用样例推断所有模型。

> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[shen-shanshan/vllm-dev-skills@1ccafeebf71b](https://github.com/shen-shanshan/vllm-dev-skills/blob/1ccafeebf71b5d922f351a0cd009a258b7474fac/skills/vllm-vs-atom-decode-trace-comparison/SKILL.md)，2026-10-31 核对。

# vLLM vs ATOM Decode Trace 对比分析

## 目标与输入

产出一份 Markdown 对比报告：同一模型、同一配置下 vLLM 与 ATOM 的 decode 阶段 GPU kernel 实现差异，
并对每类层生成对比表 + vLLM 优化 TODO 清单。

需要 4 个输入（缺失则询问用户）：

1. 模型名（如 `MiniMaxAI/MiniMax-M3`）
2. vllm trace 路径（torch profiler 导出的 Chrome-trace JSON，可 `.gz`）
3. atom trace 路径（同上）
4. 报告语言：`zh`（默认）/ `en` / `both`

第 1 步先向用户确认"两份 trace 是同模型同配置跑的"这一前提；若用户不确认，
在报告 §1 中显式标注风险。配置明显不同（GPU 数量、引擎版本差异过大）时提醒用户，不拒绝执行。

## 运行提取脚本

对两份 trace 各执行一次（产物写入本 skill 的 outputs/）：

```bash
python3 <skill_dir>/scripts/extract_trace.py --trace <vllm_trace.json> --output-dir <skill_dir>/outputs
python3 <skill_dir>/scripts/extract_trace.py --trace <atom_trace.json> --output-dir <skill_dir>/outputs
```

stdout 会给出 `digest:` 与 `slice:` 两个产物路径，以及 `engine/steps/chosen_step/kernels/distinct/replay_stable/busy_pct` 摘要。

- 脚本自动：识别引擎（顶层 key）、用 gpu 通道 `execute_context_*` 注解切分 decode step、跳过首尾窗口选中间 step、
  对每个 kernel 名做统计（count/mean/median/sum/streams）、取中间 occurrence 作单层样本、输出 digest(.md) + slice(.json)。
- 一般无需调参。仅当 digest 异常（如 `replay_stable: False` 且回退窗口可疑）时才用
  `--step-index N` / `--occurrence N` 覆盖；trace 无 gpu 注解时用 `--bracket cpu-annot`（结果标注为近似）。
- 退出码 2 = trace 没有 step 注解（需用带 profiler 注解的方式重新采集），此时停止并向用户说明。

## 加载领域知识

**必须完整阅读** `references/analysis_guide.md` 后再开始解释数据。该文件包含：
trace 格式事实、步骤切分原理、layer 识别方法（occurrence-index）、逻辑算子分组规则、
多流检测、对比表与 TODO 编写规则、pitfalls、报告模板、以及 MiniMax-M3 的完整 worked example
（含两引擎真实 kernel 清单，可作方法论参照——其它模型按同样流程处理）。

## 逐层类型对比流程

1. **校验可比性**：读两份 digest 的 §1-§3。确认引擎、rank/device 与窗口可比；原脚本 `busy` 是 kernel duration 求和，不是区间并集利用率，不能以 ~90-100% 作验收；
   replay_stable 状态一致。不一致时在报告 §6 说明。
2. **判定层型与选中层**：按 guide §3，用计数族 + kernel 模板变体 + ts 位置判定该模型的层类型分布，
   每类层选一个中间层（digest §5 的 middle-occurrence 行即该层样本）。
   若模型有 HF config 可查（网络可用时用模型名查 `num_hidden_layers`、dense/sparse 层分布），与计数族互证。
3. **归组逻辑算子**：按 guide §4 的标准 forward 序列把每类层的 kernel 归组为逻辑算子；
   对不认识的 kernel 名，用 digest §7 的全名在对应引擎仓库 grep（vllm: `csrc/`、`vllm/_custom_ops.py`；
   ATOM/aiter: `csrc/`），确认语义后再归组。
4. **生成对比表**：每类层一张表（模板见 guide §6/§8）。数字只取自 digest/slice：
   使用 `Σ(count×mean) / 该层型层数` 得到每层总 GPU 工作量；重复次数、融合和重叠需明示。5%/20%只是原报告的描述分档，不能替代重复测量方差或统计显著性。
5. **派生 TODO 清单**：按 guide §6 规则，只对 atom 更快且差异可复现的算子写 TODO（写给 vLLM），
   按预期收益排序。

## 生成报告

按 guide §8 的模板拼装最终报告：

- 文件名：`outputs/<model-slug>-trace-comparison_<zh|en>_YYYYMMDD_HHMMSS.md`
  （`model-slug` 取模型名小写去特殊字符，如 `minimax_m3`；时间戳用运行时刻）。
- 语言：`zh` 中文标题与正文（kernel 名保留英文）；`en` 全英文；`both` 生成两份。
- 所有数字必须能回溯到 digest/slice（报告 §6 注明两份 digest/slice 的文件名作为审计线索）。
- 表格中 kernel 名可用 digest 的截断名 + 短 id（如 `k27`），第一次出现时给全名。
- 完成后：保留 outputs/ 中的 digest/slice 中间产物，向用户展示报告路径与核心结论摘要。

## 不适用场景（停止并说明）

- trace 中没有 `execute_context` 注解且 `--bracket cpu-annot` 也无效（退出码 2）；
- 用户要对比的是 prefill（无 decode step 注解）或非 GPU kernel 层面的差异；
- 两份 trace 模型或关键配置不同且用户坚持要下性能结论（可做实现差异对比，但报告须显著标注不可比）。
