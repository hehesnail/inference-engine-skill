# llama.cpp 与 GGUF 推理

GGUF 是 GGML Universal File 容器，承载 tensor 数据及 tokenizer/模型元数据；文件格式与某个 Q4/Q5 的量化算法不是一回事。精度、实际量化块格式、GPU offload、CPU ISA 和 kernel 决定容量与速度。

## 构建与执行
现代源码以 CMake 配置目标后端，例如已授权 CUDA 环境可用 `cmake -B build -DGGML_CUDA=ON`、`cmake --build build --config Release -j "$JOBS"`，再检查 `build/bin/llama-cli --help` 与 `llama-server --help`。CPU/Metal/HIP 等按当前文档选后端，不复制旧 make LLAMA_CUBLAS 参数。

convert_hf_to_gguf.py 负责支持模型转换，llama-quantize 负责相应量化；imatrix 是部分方法可用的校准信息，并非所有格式都必需。保存原模型与 tokenizer revision、转换器 SHA、quant type、必要 metadata，不能只按文件后缀认定模型兼容。

## 服务与测量
上下文长度和 GPU layers 要按实际内存及模型上限配置。CLI batch size 往往是 token processing 批量，并不等于独立请求数；向 stdin 写多行也不自动形成多请求并发。prefill 与 decode 的吞吐分开记录，llama-bench 均值不能直接变成线上 p50 TTFT/ITL。

更少 bits 不保证更快：解量化、访存、CPU/GPU分工、shape 和 dispatch 会改变结果。不要复制固定“默认context512”或任意量化质量排名。Whisper 的生态与音频模型接口另查，不把它当 llama.cpp 的通用文本模型。
参见 [量化格式](../../../../references/quantization-engine-contracts.md)、[benchmark](../../../../references/benchmarking.md)。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `12-inference-serving/llama-cpp/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [optimization](references/optimization.md)
- [quantization](references/quantization.md)
- [server](references/server.md)

完整历史正文及本次具体勘误保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于追溯；运行规范使用本页与专题。
