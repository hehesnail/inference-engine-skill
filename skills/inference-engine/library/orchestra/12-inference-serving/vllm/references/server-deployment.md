# vLLM 服务部署：进程、端口与真实就绪

本页替换旧版未经支持的 metrics-port、任意多节点启动和固定健康响应示例。命令占位符来自当前任务环境；不要把它们当作新资源授权。

## 最小服务与客户端

准备好模型与兼容环境后：
~~~bash
vllm serve --help
vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT"
~~~
OpenAI SDK 使用服务的 `/v1` base URL；raw HTTP/bench 需核对它是否自行追加 endpoint，防止 `/v1/v1/completions`。chat 模型模板与请求格式匹配。服务 bind、反代、认证和可达性沿已有部署配置。

就绪检查同时关注进程退出码、启动日志、HTTP状态与模型身份。用合理超时等启动，失败时返回日志，不能无限等待。`/health` 可以是200空body；不要 grep 一个想象的JSON字段。HTTP就绪后按任务需要发一次实际生成，区分只读探针与会消耗GPU的生成探针。

## 容器、Kubernetes 与多副本

固定 image digest、模型挂载、实际参数和进程GPU可见性。两个服务改不同端口仍可能争用同一组GPU，分别记录设备绑定与资源分配。模型加载/JIT/graph可能很慢，startupProbe、readiness、liveness的时限按真实启动行为配置，不复制任意30秒/60秒阈值。

SSE反代需要保留streaming、超时和取消传播语义。每副本metrics分别采集；负载均衡后的单一metrics URL可能轮流命中不同进程，不能当成完整聚合状态。

## 指标

核对快照的 `/metrics` 与服务共享端口，Prometheus的自身9090不是vLLM服务指标端口。先抓实际样本再选择指标名称与labels，不按旧文中下划线拼名。
对于真实存在的 histogram，例如：
~~~promql
histogram_quantile(0.95, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))
~~~
跨实例聚合需保留所关心的model/instance等labels并确保bucket一致；仅对bare bucket累计值直接histogram_quantile不是窗口延迟分位数。不能用分位数减法推出两个阶段的分位数。

## 分布式

选择目标版本支持的executor与启动方式。TP×PP及DP/EP、rank、节点GPU数、rendezvous要与实际进程一致；在两台节点各启动完整TP8/PP2服务并随意设WORLD_SIZE=2不是通用部署方法。确认拓扑、模型分片和端口后用对应官方多节点示例。
