# 性能对照：先选择工作负载

离线批量评测先读 [offline-evaluation.md](offline-evaluation.md)；精度门槛和候选启用条件见 [numerical-validation.md](numerical-validation.md)。本页覆盖共同的对照原则，并补充在线服务与 kernel 的计时细节。

完整流程、结果 schema、配置模板与配对验证细节已保存在 [BBuf benchmark 指南](../library/bbuf/skills/llm-serving-auto-benchmark/GUIDE.md)。[vLLM 在线 benchmark](../library/vllm-skills/plugins/vllm-skills/skills/vllm-bench-serve/GUIDE.md)与[prefix cache 指南](../library/vllm-skills/plugins/vllm-skills/skills/vllm-prefix-cache-bench/GUIDE.md)也可离线读取。配置检查和结果比较通过[工具入口](tools.md)执行。

先界定要回答的是代码修改的收益、给定配置的性能，还是指定预算内的调优结果。代码 A/B 固定环境和参数，只换实现；部署调优可以换参数，但把每个候选及其失败记录下来。只运行用户指定的框架。

## 保持可比

用同一请求计划，保留输入记录顺序、实际执行数和筛选原因。记录模型、tokenizer/processor、chat template、精度/量化、采样/stop 条件、输入输出长度分布、并行拓扑、硬件及框架和客户端版本。相同文本或随机种子未必生成相同 token 序列；核对服务实际 usage 和有效长度。

明确负载方式：固定 request rate、并发上限和瞬时 burst 不是同一实验。记录 warmup、编译/图捕获状态、超时、失败、早停与实际输出 token 数。设定的 output length 不能当实际完成长度。

先用小规模检查运行是否有效，再按要回答的问题选择基线；用户关注整机离线吞吐时，单请求或单卡结果只能作辅助，不能替代目标队列、batch 和并行拓扑。需要寻找最佳配置时，在授权时间内围绕有依据的参数调整；声明搜索范围，不能只跑一个对手默认配置便声称全局最优。OOM 或不支持的候选保留原因，已知无效配置不盲目重试。

逻辑评测 batch、前端待提交数量、引擎活跃请求上限和每步 token budget 分别记录。固定一批入队与持续补充请求，即使采样参数和 seed 一致，也可能改变实际算子形状与输出轨迹；分数变化先做匹配调度的对照，不能直接归为随机波动。聚合吞吐不能当作每请求 decode 速度，多卡套件加速不能直接当作 B1 单卡收益。

## 原生工具入口

vLLM 通常使用 `vllm bench serve`；需要可用的参数扫描工具时查当前 `vllm bench sweep serve`。SGLang 查目标版本的 serving benchmark 模块；此次参考源使用 `python -m sglang.benchmark.serving`，旧版本可能不同。不要在目标环境里为了符合此文件先升级依赖。

下面是 vLLM OpenAI chat 客户端的命令形状，先用目标版本帮助确认参数，再填写本次值。`BASE_URL` 指主机根地址，endpoint 单独指定，避免重复 `/v1`。

```bash
vllm bench serve --backend openai-chat \
  --base-url "$BASE_URL" --endpoint /v1/chat/completions \
  --model "$SERVED_MODEL" --dataset-name random \
  --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
  --num-prompts "$N" --request-rate "$RPS" --max-concurrency "$CONCURRENCY" \
  --save-result --result-dir "$RUN_DIR"
```

合成数据适合流程检查与可控形状实验；对真实流量的结论使用相应请求分布。比较框架时优先同一客户端版本和协议；用不同原生客户端则先核对计时口径、模板处理、token 计数和错误统计。

## Prefix cache 单独控制

将以下条件分开：关闭缓存、启用但冷缓存、按固定顺序预热后复用。每个候选的起始状态必须一致；warmup 可能同时预热 kernel 与 prefix，二者分别记录。

固定共享前缀 token、独特后缀、重复次数、请求顺序及路由目标。多 worker 下请求是否到同一个缓存域也是条件。块边界、部分命中、淘汰或缓存容量压力只按本次问题选取。缓存 reset 仅用于本次拥有的隔离服务，不能冲掉共享线上实例的缓存。

官方 `vllm-prefix-cache-bench` 提供 offline benchmark 和在线 `prefix_repetition` 两类入口；先确认当前脚本/参数存在。离线结果不包含完整 HTTP 调度和输出开销。

命中率、缓存 token 统计和延迟分别验证；一个 usage 数值变化不能单独证明性能改善或缓存实现正确。

## 真实 agentic 回放

使用原始完整 prompt 的 token IDs 核对共同前缀；字符级重叠或分段 tokenize 后拼接不能替代。若输入已按训练格式渲染，不再叠加 chat template。前缀重合比例只是潜在复用量，实际 cache hit、命中 token 和耗时仍需从运行验证。

说明下一轮历史使用录制答案还是本次生成答案，是否真的执行工具。录制轨迹回放可比较相同输入的缓存与性能，但不证明生成答案分支或完整 agent 闭环已覆盖。短时显存观察、代表请求、完整多轮轨迹分别标注执行范围；抽样不能写成全量。

## 解释结果

- TTFT、TPOT/ITL、端到端延迟分开说明。说明 TTFT 是否包含客户端排队、连接与非 token chunk；单 token 响应的 TPOT 可能未定义。
- throughput 与满足指定 SLO 的 goodput 分开。使用用户需要的分位数，不能用 p50 通过代替 p99 达标。
- 保存启动命令、客户端命令、原始结果、版本及失败信息。报告试过的配置、波动与适用范围；重复次数根据噪声和结论需要确定。
- 有 GPU 窗口时，单项结束只检查是否有效、失败/OOM、长度与配置是否明显异常；有效且下一项已准备好就继续。详细归因与绘图后置。

kernel 微基准先核对数值、shape 和真实调用路径，再计时。编译、分配、拷贝等是否计入由问题决定并保持对称；生产 wrapper 的必要同步不能只从候选侧剔除。说明 warm/cold cache 和图模式。多 rank 延迟按同一轮中最慢 rank 统计，不能拿 rank0 当整体。

未经 profiling 的正式计时用于性能结论；trace 用于定位瓶颈。微基准、模型 forward 和在线服务结果分别报告。初始化、编译／预热、推理及 CPU 判分分列；两版本均预热后再解释稳定性能。整批输出吞吐使用实际输出 token 除以共同整批窗口，并说明是否包含 tokenize、prefill、decode 或排队；不能将它标作纯 decode 吞吐。套件总耗时比用可比任务的参考耗时之和除以候选耗时之和，不平均单项加速比。

## 多轮优化的记录

需要比较多个实现或参数候选时，用 [优化工作流](optimization-workflow.md) 区分原始基线、当前最好版本、候选与最终生效配置。kernel 局部计时与端到端收益的接线见 [kernel-engine-integration.md](kernel-engine-integration.md)。
