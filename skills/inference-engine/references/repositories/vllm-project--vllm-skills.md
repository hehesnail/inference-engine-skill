# vllm-project/vllm-skills

固定提交：[c99623410c1531148ff9b39fe1dcb27efbf4bf23](https://github.com/vllm-project/vllm-skills/tree/c99623410c1531148ff9b39fe1dcb27efbf4bf23)。

先读本地适配：[部署修正](../deployment-verified.md)。

当前采用 6/6 项。采用项正文逐篇语义审阅；相关源码和附属资源的核对范围单列，不能推广成整仓每行已验证。技术修订与限制见 [语义核对](../semantic-audit.md)。

| 专项 | 用途 |
|---|---|
| [vllm-bench-random-synthetic](../../library/vllm-skills/plugins/vllm-skills/skills/vllm-bench-random-synthetic/GUIDE.md) | 在线随机请求smoke/吞吐TTFT-TPOT测量；明确只停止自己启动的服务。 |
| [vllm-bench-serve](../../library/vllm-skills/plugins/vllm-skills/skills/vllm-bench-serve/GUIDE.md) | 按负载率、并发、数据集和SLO配置服务压测，能生成JSON与逐请求证据。 |
| [vllm-deploy-docker](../../library/vllm-skills/plugins/vllm-skills/skills/vllm-deploy-docker/GUIDE.md) | NVIDIA Docker镜像/源码构建、预编译wheel、共享内存和模型缓存挂载的基本路线。 |
| [vllm-deploy-k8s](../../library/vllm-skills/plugins/vllm-skills/skills/vllm-deploy-k8s/GUIDE.md) | Deployment/Service/probe/端口和TP-GPU配对，适用明确要求K8s部署。 |
| [vllm-deploy-simple](../../library/vllm-skills/plugins/vllm-skills/skills/vllm-deploy-simple/GUIDE.md) | 已用deployment-verified替代有问题quickstart，保留启动身份、HTTP2xx、模型列表与真实生成契约。 |
| [vllm-prefix-cache-bench](../../library/vllm-skills/plugins/vllm-skills/skills/vllm-prefix-cache-bench/GUIDE.md) | 离线引擎与在线prefix_repetition两条缓存性能实验路线，能保持输入条件对照。 |
