# intel/gpu-ai-skills

固定提交：[d0f47fcf262560b06fea1684c70d80c877e04854](https://github.com/intel/gpu-ai-skills/tree/d0f47fcf262560b06fea1684c70d80c877e04854)。

先读本地适配：[XPU接线](../backend-kernel-development.md)。

当前采用 22/23 项。采用项正文逐篇语义审阅；相关源码和附属资源的核对范围单列，不能推广成整仓每行已验证。技术修订与限制见 [语义核对](../semantic-audit.md)。

| 专项 | 用途 |
|---|---|
| [cuda-to-xpu-migration](../../library/intel/plugins/intel-gpu-ai-skills/skills/cuda-to-xpu-migration/GUIDE.md) | CUDA依赖清单区分Python迁移、native库、服务HTTP provider替换。 |
| [enable-xpu-monitoring](../../library/intel/plugins/intel-gpu-ai-skills/skills/enable-xpu-monitoring/GUIDE.md) | XPU硬件metrics与vLLM指标联合观测，localhost服务与现有栈复用。 |
| [llamacpp-xpu-run](../../library/intel/plugins/intel-gpu-ai-skills/skills/llamacpp-xpu-run/GUIDE.md) | 固定llama.cpp SYCL tag、device selector和GGUF基线可辅助后端对比。 |
| [model-can-it-fit](../../library/intel/plugins/intel-gpu-ai-skills/skills/model-can-it-fit/GUIDE.md) | 配置推导weights/KV/SSM/vision分项与假设，避免单纯参数量估算。 |
| [model-config-recommend](../../library/intel/plugins/intel-gpu-ai-skills/skills/model-config-recommend/GUIDE.md) | roofline候选+校准+实测三级以及DP/TP布局、模型revision对齐。 |
| [sglang-xpu-bench](../../library/intel/plugins/intel-gpu-ai-skills/skills/sglang-xpu-bench/GUIDE.md) | 在线SGLang TTFT/TPOT/ITL与原始JSONL、唯一运行文件和量化输出smoke。 |
| [sglang-xpu-run](../../library/intel/plugins/intel-gpu-ai-skills/skills/sglang-xpu-run/GUIDE.md) | XPU image/UMD、attention backend、page size、TPcollective探针与真实内容检查。 |
| [torch-xpu-bench](../../library/intel/plugins/intel-gpu-ai-skills/skills/torch-xpu-bench/GUIDE.md) | 独立torch generate计时和XPU同步、warmup、实际device检查。 |
| [torch-xpu-profile](../../library/intel/plugins/intel-gpu-ai-skills/skills/torch-xpu-profile/GUIDE.md) | Kineto XPU/CPU timeline、schedule和源op关联用于定位等待与算子热点。 |
| [torch-xpu-run](../../library/intel/plugins/intel-gpu-ai-skills/skills/torch-xpu-run/GUIDE.md) | torch.xpu namespace与xccl迁移、wheel ABI配对、logical device核对。 |
| [vllm-omni-xpu-run](../../library/intel/plugins/intel-gpu-ai-skills/skills/vllm-omni-xpu-run/GUIDE.md) | 多stage XPU资源预算、实际deploy YAML与stage override生效核对，多模态内容验证。 |
| [vllm-xpu-bench](../../library/intel/plugins/intel-gpu-ai-skills/skills/vllm-xpu-bench/GUIDE.md) | 同口径serve/throughput、EOS与缓存条件、原始客户端请求指标。 |
| [vllm-xpu-profile](../../library/intel/plugins/intel-gpu-ai-skills/skills/vllm-xpu-profile/GUIDE.md) | server和offline profiler，逐rank时间线与实际attention/quant dispatch对照。 |
| [vllm-xpu-run](../../library/intel/plugins/intel-gpu-ai-skills/skills/vllm-xpu-run/GUIDE.md) | XPU image入口、设备发现、quant/KV/attention组合和实际generation验证。 |
| [xpu-container-run](../../library/intel/plugins/intel-gpu-ai-skills/skills/xpu-container-run/GUIDE.md) | DRM节点/GID、共享内存与容器entrypoint是XPU runtime接线必需。 |
| [xpu-deploy-plan](../../library/intel/plugins/intel-gpu-ai-skills/skills/xpu-deploy-plan/GUIDE.md) | 把preflight/fit/config/launch/smoke汇成可审阅部署计划。 |
| [xpu-discover](../../library/intel/plugins/intel-gpu-ai-skills/skills/xpu-discover/GUIDE.md) | XPU inventory、kernel driver与sensor证据缺失区分，版本化CLI。 |
| [xpu-model-type-detect](../../library/intel/plugins/intel-gpu-ai-skills/skills/xpu-model-type-detect/GUIDE.md) | 读config/模型卡选择非文本loader，避免所有模型套CausalLM。 |
| [xpu-port](../../library/intel/plugins/intel-gpu-ai-skills/skills/xpu-port/GUIDE.md) | libcst迁移与semantic/escalate区分，CPU FP64对照作为前向sanity。 |
| [xpu-profile-unitrace](../../library/intel/plugins/intel-gpu-ai-skills/skills/xpu-profile-unitrace/GUIDE.md) | SYCL/LevelZero提交、拷贝和collective追踪补充torch op视图。 |
| [xpu-runtime-preflight](../../library/intel/plugins/intel-gpu-ai-skills/skills/xpu-runtime-preflight/GUIDE.md) | read-only主机与可选image preflight明确执行尚未验证，保留原始日志。 |
| [xpu-system-setup](../../library/intel/plugins/intel-gpu-ai-skills/skills/xpu-system-setup/GUIDE.md) | Intel driver/userland/firmware分层定位，以及官方OMIX文档与live差异检查。 |

## 仅做范围筛选的入口

下列条目只核对名称与描述，未全文语义审阅，不认定无用；完整源码留在固定checkout，任务涉及其领域时再读。

| 来源入口 | 本轮范围 |
|---|---|
| [template/SKILL.md](https://github.com/intel/gpu-ai-skills/blob/d0f47fcf262560b06fea1684c70d80c877e04854/template/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
