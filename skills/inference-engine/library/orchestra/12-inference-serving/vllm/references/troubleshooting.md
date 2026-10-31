# vLLM 排障：按失败阶段取证

先冻结版本、安装位置、模型/tokenizer、输入、命令与必要日志，明确错误发生在下载、加载、profile、graph capture、调度、执行、输出还是客户端。不以无关参数组合盲试替代定位。

## 内存
- 权重加载OOM：查实际weight dtype、分片、双副本和临时conversion。降低KV预算不必然能加载超过物理容量的权重；量化需真实兼容格式。
- profile/capture OOM：查batch/tokens、workspace、graph sizes与保留池。缩小KV预算可能留出更多非KV空间，但需与实际阶段对应。
- 请求期OOM：查真实shape、临时buffer、并发、state/encoder cache和其他进程。KV池满常由调度机制处理，不能把所有OOM归因“KV填满”。
- 缓存容量不足：核对max_model_len与可分配blocks，降低上下文或增加资源需符合任务要求。

## 吞吐与延迟
util低、VRAM满或固定req/s不足不直接判memory-bound。沿queue→CPU/host→GPU kernels→通信→输出测量，必要时用trace与NCU。TTFT、ITL、TPOT、请求端到端和goodput分开。先核对错误/取消和实际输出长度，再对照图、batch、cache及backend。

TP非通用2的幂限制；维度/heads/kernel与拓扑才决定支持。prefix命中变化不等同单组件时间收益。

## 加载、接口与分布式
模型not found检查ID、revision、权限、挂载和网络；`trust_remote_code`仅在确需该模型自定义代码时使用。缺FlashAttention先查引擎自身backend依赖和包兼容，不编造`--disable-flash-attn`作为万能开关。
量化质量差先比固定IDs/模板、权重、scale与中间输出，不靠提高temperature修复。

WORLD_SIZE按进程组定义，不一律等于节点数。`NCCL_P2P_DISABLE=1`是禁用P2P，不是“强制P2P”。超时配置核对实际框架/版本，不能假设任意`NCCL_TIMEOUT`有效。NIC名与端口来自现场配置，不默认eth0。网络问题不自动授权开放防火墙。

## 记录与收尾
bash日志管道使用pipefail保留服务退出状态。诊断只包含必要配置，隐藏token/凭据；函数级trace等高开销debug关闭后再测性能。收集实际`/metrics`输出、健康HTTP状态和生成路径，不假定9090或虚构指标名。停止本任务的进程并保留复现输入与修复diff。
