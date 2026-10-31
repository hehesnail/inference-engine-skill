# 部署与就绪验证

用于把官方 vLLM 简易部署指南转成当前环境可执行的步骤。完整 Docker/K8s 模板仍在 [catalog](catalog.md)。先复用本任务指定的环境和模型；需要新环境时按实际硬件、Python/驱动和目标版本的安装矩阵准备，不凭 nvidia-smi/gcloud 是否存在猜测安装包。

## 环境和启动

1. 在执行环境记录 Python 路径、包版本、vllm.__file__、源码 HEAD/dirty 或镜像 digest。核对 vllm serve --help 中实际支持的参数。不能以维护机的包代替容器里的包。
2. 使用任务选定的 MODEL、HOST、PORT、设备和上下文长度。示例命令的变量必须由当前环境填写，不使用默认模型下载来完成健康检查。
3. 以现有服务管理器、已命名容器或本任务保存的进程句柄启动。记录启动命令与日志；清理前核对该句柄、创建时间和命令身份，不能只相信可能复用的 PID 文件。
4. 先判断 worker 是否仍存活，再检查 readiness。HTTP 传输成功与 HTTP 2xx、JSON 结构、真实生成是四件事。

```bash
# 对已有且获准访问的服务执行；不是离线工具。
curl --fail --show-error --silent --max-time 10 "$BASE_URL/health"
curl --fail --show-error --silent --max-time 10 "$BASE_URL/v1/models"
# BODY_FILE 由当前任务准备，包含实际 served model 名和最小有效请求。
curl --fail --show-error --silent --max-time 60 \
  -H 'Content-Type: application/json' --data-binary "@$BODY_FILE" \
  "$BASE_URL/v1/chat/completions"
```

检查模型列表非空、served name 与请求匹配，生成结果具有所需 choices/content 或工具调用结构。若目标本来允许空文本或早停，按该任务的输出契约判断，不机械要求固定 token 数。PD 场景还需真实 P→router→D 生成证明 KV 交接，/health 200 不足以证明。

## Docker / K8s

固定镜像、显式设备范围；共享内存和 KV 容量分别判断。K8s startup probe 预算覆盖加载/编译，readiness 决定收流，liveness 不能把正常长初始化变成重启循环。GPU requests 与 TP/PP/DP 组匹配，namespace、Service selector、containerPort、targetPort、探针端口和服务参数一致。滚动更新遵循当前任务要求。

模型缓存、既有环境、其他服务不因为示例 cleanup 而删除。失败时保留本次命令与日志，区分依赖/加载/端口/初始化/worker 退出/API 内容错误，再针对具体原因修复。

## 原 quickstart.sh 的地位

[原脚本](../library/vllm-skills/plugins/vllm-skills/skills/vllm-deploy-simple/scripts/quickstart.sh) 保留供追溯，GUIDE 已取消把它作为默认执行路径。它没有可靠的后端安装分流、HTTP 状态检查和进程身份检查；不能将其 start 或 status 输出当作就绪证据。
