# vLLM 量化：确认格式、加载器和 kernel

算法名不决定当前引擎是否支持。读取实际安装版本的 quantization registry、模型config与加载日志，确认所用路径；查到同名上游tag不替代被测安装。

## 支持与配置
AWQ/GPTQ预量化权重需要匹配group、zero point、scales、packing、desc_act/g_idx和后端限制。`--quantization awq` 不会普遍把任意FP16模型变成AWQ。GPTQ的activation order在checkpoint/quantization config中定义，旧文`--gptq-act-order`不是所核对serve参数。

固定7b77ed5 registry没有内置hqq/bitsandbytes，其他版本或注册插件另行确认。FP8支持不是“H100 only”的永久规则，也不是所有模型都能在线直接转换；硬件、格式、scale与kernel实际支持共同决定。选择dtype不等于完成量化。

## 容量
70B4bit原始权重约35GB，FP16约140GB（十进制），这些不包括metadata、未量化层、KV/state、临时和图池。不能据此保证40GB单卡AWQ服务可运行。加载峰值与稳态内存分开，量化过程也可能需要原权重和激活。

## 校准
校准数据覆盖目标分布，保存精确IDs/mask与长度，而非随便写128–512样本即认为代表性足够。AutoGPTQ核对版本需要样本字典`input_ids`与`attention_mask`；先构造配置再from_pretrained，原版先用未定义quantize_config的顺序已撤回。AutoAWQ属于已deprecated的历史路线，保持复现环境或按当前支持选工具。

## 验收
固定输入与质量指标方向：accuracy下降与perplexity上升都可能退化，明确绝对百分点还是相对变化、零基线与非有限值处理。问一个“量子纠缠”问题不能验证量化质量。
删除原指南无来源的99%/98.5%、1.8×和“production-ready”表。实际交付同时记录任务质量、内存、prefill/decode/端到端及失败数，不能从bit宽推断速度。
