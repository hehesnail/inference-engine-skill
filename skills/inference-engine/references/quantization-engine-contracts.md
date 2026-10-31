# 量化与稀疏权重接入推理引擎

用于 AWQ/GPTQ/HQQ/bitsandbytes/GGUF、FP8/FP4 或稀疏权重的加载、执行与性能问题。保留各方法的具体差异，先核对 checkpoint 与当前 loader，再考虑 kernel。

## 四层契约

| 层 | 必须对应的内容 | 常见误判 |
|---|---|---|
| 数学/校准 | 位宽、对称性、group axis/size、clip/round、校准样本与目标分布 | 把无校准等同无质量损失 |
| 文件/布局 | quant_method、packed weight、scale/zero、g_idx、block shape、permute、未量化层 | 给 FP16 模型加 AWQ 参数就当成完成量化 |
| kernel/硬件 | 激活与累加 dtype、对齐/shape、ISA、反量化、repack、fallback | 文件更小就推断更快 |
| 服务 | KV/state/workspace/graph、TP/EP 分片、batch、上下文、LoRA、质量 | 仅凭权重体积判断整个模型可部署 |

内存下界：70B 参数全为 4-bit 已需约 35 GB 十进制空间；尚未计 scale/zero、非量化 tensor、KV、workspace 和运行时。因此“70B 4-bit 单卡24GB”不能当作全驻留方案。若依赖 CPU/disk offload，必须说明真实 placement、传输和容量。分片还需满足 head/expert/packed-block 的约束，不能只用总显存相除。

## 方法的可用知识与限制

- **AWQ**：用激活统计指导通道尺度变换，降低显著权重的量化误差。不能理解成固定保存1%权重为FP16。GEMV 与 GEMM 的优势取决于 token batch；Marlin 是特定 packing/kernel 路径，不能仅写字符串保证转换成功。AutoAWQ 来源已声明弃用；已有模型可继续在兼容环境加载，新量化按当前维护工具与目标引擎格式选择。
- **GPTQ**：利用近似二阶信息逐步补偿量化误差。group size 越小，scale/zero 等元数据通常越多；-1 表示该实现不作小组划分，不是“最高精度且最慢”的通用配置。scale 并不因为权重4-bit就也为4-bit。act-order/g_idx 与 kernel 支持、权重重排要一起核对。
- **HQQ**：量化算法本身可以不使用校准数据，性能仍依赖优化后的执行后端。核对的 Quantizer 对 axis=1 用 `reshape(-1, group_size)`，axis=0 用 `reshape(group_size, -1)`；不能简写成“输入维/输出维”后忽略 reshape。HQQLinear 的 scale/zero 在 meta 中，且可能再次量化；不要假设有独立 .scale 属性。
- **bitsandbytes**：量化权重 dtype 与 compute dtype 分开。NF4 是表示形式，double quant 是再压缩量化常数，不能当“额外精度”开关。LLM.int8 的 outlier 路径与 CPU FP32 offload 影响容量和成本。offload_folder 不会自动把所有超额显存转成可用的磁盘执行。
- **GGUF**：GGML Universal File 是容器与元数据格式，非单一量化算法；每个 tensor 的实际类型、有效 bits/weight 和 loader 支持需核对。imatrix 用代表性输入估计重要性，不是所有 Q4 模型的强制前提。生成模型时保留 tokenizer、special tokens、chat template 和 tensor 元数据；Python binding 与 native llama.cpp 也有独立构建版本。

量化后的质量/速度不能用 AWQ>GPTQ>bnb 或“位数越低越快”的固定表预测。低位宽可能增加 unpack/repack、scale 和 padding 成本；高并发 compute-bound 与低并发 bandwidth-bound 的收益不同。

## 已核对的 API 反例

核对 vLLM `7b77ed581d92` 的 quantization registry 没有原生 `hqq` 或 `bitsandbytes` 条目；原 Orchestra HQQ 的 `LLM(quantization="hqq", quantization_config=...)` 不能直接用于该版本。其他版本/插件可能支持，查询当前 registry、加载日志和实际 method，不能由 Transformers 支持推断 vLLM 支持。

核对 HQQ core `HQQLinear.set_backend` 接受 `HQQBackend` 枚举并读取 .value；不能传 `"marlin"` 或 `"torchao_int4"` 把转换后的优化模块当成核心枚举。使用该版本官方 backend conversion 接口，并重新核对最终 module 类型。

AutoGPTQ 原接口的校准输入是含 input_ids 和 attention_mask 的字典，不能传裸 token list；quantize 的签名没有 use_marlin。旧代码复现保留其依赖组合；新任务先核对维护状态，避免为了教程换掉被测环境。

[实际 vLLM registry](https://github.com/vllm-project/vllm/blob/7b77ed581d92fa14e34ab43344837bb1ca1b4257/vllm/model_executor/layers/quantization/__init__.py)、
[HQQ 实现](https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py)、
[AutoGPTQ 校准入口](https://github.com/AutoGPTQ/AutoGPTQ/blob/main/auto_gptq/modeling/_base.py)。

## 稀疏不是把密集 tensor 置零

Wanda 的统计需在所有校准 token 上累积输入通道的 L2 信息，按输出行的权重重要性选择，不是覆盖为最后一个样本的 abs.mean(dim=0)。SparseGPT 的核心操作对象是层，不能把整个 AutoModel 传入后假设有统一 .prune()。

零值仍存储在普通密集权重中时，文件和 GEMM 不会自动减半。2:4 加速需要特定维度、dtype、shape、压缩元数据及真正的 sparse kernel；任意4:8也不等于硬件支持。结构化删除 head/neuron/layer 必须同步修改邻接投影、残差、配置、分片及 loader。训练恢复时掩码是否保持，决定零权重会不会重新长回来。

## 验证一个接入或优化

选择本任务影响的路径：加载与反量化小样、真实 module/dispatch、相关 shape/TP 分片、固定 token 的 logits 或中间层、任务质量与同 workload 性能。质量指标先确定方向和单位：accuracy 下降0.05绝对值与相对下降5%不同；PPL 越低越好。零基线、NaN/Inf、缺失指标不能被截断成“无退化”。

质量通过后再判断性能收益；保存转换参数、校准数据身份和 checkpoint revision。更细的候选流程见 [优化工作流](optimization-workflow.md)。
