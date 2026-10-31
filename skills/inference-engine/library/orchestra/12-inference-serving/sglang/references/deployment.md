# SGLang 部署：版本、就绪与分布式

固定服务的源码/镜像和模型身份，检查 `python -m sglang.launch_server --help`。基础命令用实际model-path、host与port，TP/量化/图模式按模型及硬件支持选择。

## 环境与探针
旧CUDA12.1+torch2.4 FlashInfer索引不是通用当前安装路线；已有容器先看实际依赖与加载文件。Docker CMD的固定参数不会因设置MODEL_PATH/TP_SIZE环境变量自动替换，需要明确entrypoint或传入command。

健康探针检查状态码、进程与日志。`/health`可返回200空body；`/health_generate`会触发实际生成，不属于零GPU只读探针。startup/readiness/liveness按真实加载时间配置，失败后保留具体异常。

## 指标与路由
启用实际支持的metrics配置后抓取`/metrics`，根据真实名字、labels与单位建立面板。旧文`/stats`、固定radix_cache_hit_rate等字段不可假定存在；不能靠编造指标求>70%门槛。
多副本单独采集指标，反代SSE保留stream、timeout与取消传播。cache-aware路由的收益需和真实prefix分布、queue及负载平衡一起测。

## 容量与多节点
`mem_fraction_static`涵盖模型权重和cache等static内存预算，不是“全部给KV的GPU比例”。所核对版本的容量字段包括max_total_tokens；旧`--max-radix-cache-len`不是通用选项。max_running_requests影响调度/临时内存，调整需固定workload。
多节点使用对应版本的nnodes/node-rank/dist-init-addr等接口并对齐总TP与各节点GPU数；不是Ray启动后加一个`--num-nodes`就必然完成部署。

## 成本与运维
实例价格与吞吐从当前测量取得，token/s转每小时需乘3600；旧文1000tok/s=1Mtok/hour的计算错误已撤回。扩容按真实延迟/队列和可用指标设计，不把固定GPU利用率80%变成所有服务标准。
凭据、对外发布和资源范围来自当前部署任务，不在示例里硬编码密钥/服务器。
