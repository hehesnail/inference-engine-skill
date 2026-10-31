# vLLM：部署、缓存、量化与性能定位

先区分 checkout、安装包与运行容器，记录实际版本、模型/tokenizer revision、修改和启动参数。源码定位与官方模型接入从 [vLLM 工作台](../../../../references/vllm.md) 进入。

## 可运行的基本接口
在已准备且获授权的环境中，先查看 `vllm serve --help`。最小单服务命令形态为 `vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT"`。资源绑定、模型名和端口来自任务配置。OpenAI client 的 base_url 通常以 `/v1` 结尾；bench 客户端若自行追加 endpoint，不再重复该前缀。

HTTP `/health`、`/v1/models` 和实际生成验证覆盖不同层。固定源码快照的 `/metrics` 挂在服务端口；没有可通用复制的 `--enable-metrics --metrics-port 9090` 组合。启动日志与实际响应决定可用指标。

## 选择配置
- KV、状态、临时 activation、CUDA Graph、权重分开算。模型加载时权重已 OOM，降低 KV budget 不必然有效；量化须提供支持的 checkpoint/格式或确有实现的在线路径。
- TP 受模型维度、KV heads、量化 kernel 与拓扑约束，不是通用“必须2的幂”。PP/DP/EP 与启动进程数按实际 executor 核对。
- prefix caching/chunked prefill 是否启用看实际配置。prefix token 复用率、TTFT 与吞吐不同；冷/热初态和 eviction 单独记录。
- speculative 配置按目标版本。所核对新快照使用 `--speculative-config`，不要复制旧 `--speculative-model` 示例；看 [接受与回滚](../../../../references/speculative-decoding-contracts.md)。
- 量化看 [格式到 kernel 的契约](../../../../references/quantization-engine-contracts.md)，不以 AWQ flag 将任意 FP16 权重转换成 AWQ。

## 证据与故障
固定 workload，统计实际 tokens、失败/超时和 goodput；不以 GPU util>80% 或固定 req/s 作为通用健康标准。模型不存在先查 ID、权限、revision 和挂载，`trust_remote_code` 不是下载/权限故障修复。数值异常不能靠提高 temperature 验收。

配套 [部署](references/server-deployment.md)、[优化](references/optimization.md)、[量化](references/quantization.md)、[排障](references/troubleshooting.md) 已按本次源码契约重写，历史无出处的速度表与伪参数不再作为指令。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `12-inference-serving/vllm/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [optimization](references/optimization.md)
- [quantization](references/quantization.md)
- [server-deployment](references/server-deployment.md)
- [troubleshooting](references/troubleshooting.md)

完整历史正文及本次具体勘误保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于追溯；运行规范使用本页与专题。
