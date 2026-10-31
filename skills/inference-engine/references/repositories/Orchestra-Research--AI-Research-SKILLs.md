# Orchestra-Research/AI-Research-SKILLs

固定提交：[773a52944ba4747a18bd4ae9ade53fff041adcbc](https://github.com/Orchestra-Research/AI-Research-SKILLs/tree/773a52944ba4747a18bd4ae9ade53fff041adcbc)。

先读本地适配：[量化、推理、评测](../inference-optimization-cases.md)。

当前采用 33/98 项。采用项正文逐篇语义审阅；相关源码和附属资源的核对范围单列，不能推广成整仓每行已验证。技术修订与限制见 [语义核对](../semantic-audit.md)。

| 专项 | 用途 |
|---|---|
| [0-autoresearch-skill](../../library/orchestra/0-autoresearch-skill/GUIDE.md) | Hypothesis→controlled experiment→measure→interpret cycle, rejectedattempts and resume state valuable optimizerworkflow. |
| [mamba](../../library/orchestra/01-model-architecture/mamba/GUIDE.md) | Selective recurrent state + causal conv; full-sequence/chunked/single-step equivalence and cache lifecycle are valuable for hybrid engine support. |
| [rwkv](../../library/orchestra/01-model-architecture/rwkv/GUIDE.md) | State carry between chunks and prefilling/one-token decoding are directly useful for recurrent engine validation. |
| [huggingface-tokenizers](../../library/orchestra/02-tokenization/huggingface-tokenizers/GUIDE.md) | Pipeline stages + offsets + postprocessing + saved tokenizer/transformers bridge useful for engine discrepancy localization. |
| [sentencepiece](../../library/orchestra/02-tokenization/sentencepiece/GUIDE.md) | Normalization/whitespace marker/model, control IDs, sampling versus deterministic decoding are useful parity checklist. |
| [miles](../../library/orchestra/06-post-training/miles/GUIDE.md) | 低精度导出、routing replay、draft drift、partial rollout版本记录 |
| [slime](../../library/orchestra/06-post-training/slime/GUIDE.md) | async data buffer、版本归属、权重同步、colocation与引擎状态 |
| [verl](../../library/orchestra/06-post-training/verl/GUIDE.md) | HybridFlow控制/训练/rollout边界、LoRA同步、mask/logprob与权重版本 |
| [lambda-labs](../../library/orchestra/09-infrastructure/lambda-labs/GUIDE.md) | Ephemeral vs persistent filesystem and remote launch lifecycle useful reproductions. |
| [modal](../../library/orchestra/09-infrastructure/modal/GUIDE.md) | Container lifecycle/modelload vs warmrequest, volume persistence, concurrencyvsdynamicbatching useful remote execution. |
| [skypilot](../../library/orchestra/09-infrastructure/skypilot/GUIDE.md) | Managedjob recovery vs application checkpoints, filemounts, serve readiness andconsistentresource alternatives useful. |
| [awq](../../library/orchestra/10-optimization/awq/GUIDE.md) | Activation-aware scaling, group metadata, GEMM/GEMV batch regimes, calibration represent useful quantization choices. |
| [bitsandbytes](../../library/orchestra/10-optimization/bitsandbytes/GUIDE.md) | Quantized weight versus compute dtype; NF4/double quant and int8 outliers; loading diagnostics useful. |
| [flash-attention](../../library/orchestra/10-optimization/flash-attention/GUIDE.md) | IO-aware tiling, SDPA vs standalone layouts, MQA/window and dispatch verification are central engine optimization capabilities. |
| [gguf](../../library/orchestra/10-optimization/gguf/GUIDE.md) | Conversion→imatrix→quantize→load workflows, metadata and GPU offload parameters useful. |
| [gptq](../../library/orchestra/10-optimization/gptq/GUIDE.md) | Hessian-informed reconstruction, calibration, groups/scales/zero/act-order and supported kernel packing useful. |
| [hqq](../../library/orchestra/10-optimization/hqq/GUIDE.md) | Calibration-free weight optimization, mixed-bit choices and backend conversion valuable. |
| [bigcode-evaluation-harness](../../library/orchestra/11-evaluation/bigcode-evaluation-harness/GUIDE.md) | Separate generation artifacts and sandboxed scoring, paired task IDs/seeds/stop settings, pass@k sampling. |
| [lm-evaluation-harness](../../library/orchestra/11-evaluation/lm-evaluation-harness/GUIDE.md) | Frozen task revision, fewshot/template/tokenizer + backend HF/vLLM comparison, sample logs, quality metrics. |
| [nemo-evaluator](../../library/orchestra/11-evaluation/nemo-evaluator/GUIDE.md) | Evaluation orchestration separates endpoint deployment from harness/container jobs and exports; useful for serving eval. |
| [llama-cpp](../../library/orchestra/12-inference-serving/llama-cpp/GUIDE.md) | GGUF conversion, CPU/GPU offload, grammar, CLI/server separate workloads provide useful cross-engine reference. |
| [sglang](../../library/orchestra/12-inference-serving/sglang/GUIDE.md) | Prefix reuse, structured decoding, multimodal and request batching are useful engine workloads. |
| [tensorrt-llm](../../library/orchestra/12-inference-serving/tensorrt-llm/GUIDE.md) | In-flight batching, KV paging, TP/PP/EP, disaggregation provide engine comparison axes. |
| [vllm](../../library/orchestra/12-inference-serving/vllm/GUIDE.md) | Paged KV + continuous batching, serving/offline workflows; preserve their use, but runtime and model metadata determine command/config. |
| [guidance](../../library/orchestra/16-prompt-engineering/guidance/GUIDE.md) | Token healing, grammar masking, branch/state structured prompts can probe engine grammar contracts. |
| [instructor](../../library/orchestra/16-prompt-engineering/instructor/GUIDE.md) | Pydantic postvalidation/retry and partialstream help distinguish client behavior from engine grammar. |
| [outlines](../../library/orchestra/16-prompt-engineering/outlines/GUIDE.md) | Schema compilation, request-local grammar state and batch/state resets useful constrained decoding. |
| [llava](../../library/orchestra/18-multimodal/llava/GUIDE.md) | Vision preprocessor→projector→image token expansion→LLM and conversation template alignment are engine integration anchors. |
| [stable-diffusion](../../library/orchestra/18-multimodal/stable-diffusion/GUIDE.md) | Separate text encoding/iterative denoising/VAE, scheduler and adapter state, batching/offload conditions useful SGLang diffusion. |
| [whisper](../../library/orchestra/18-multimodal/whisper/GUIDE.md) | Encoder/decoder, language/task control, timestamp and segmented output contracts useful audio engine support. |
| [long-context](../../library/orchestra/19-emerging-techniques/long-context/GUIDE.md) | RoPE pair layout/position offsets/scaling, ALiBi/window and KV memory provide long-context backend parity. |
| [model-pruning](../../library/orchestra/19-emerging-techniques/model-pruning/GUIDE.md) | Sparsitypattern, calibrationstatistics, supported sparsekerneldispatch andpackformat useful. |
| [speculative-decoding](../../library/orchestra/19-emerging-techniques/speculative-decoding/GUIDE.md) | Draft/verify/accepted tokens vs target steps, rollback and speedups under workload are essential. |

## 仅做范围筛选的入口

下列条目只核对名称与描述，未全文语义审阅，不认定无用；完整源码留在固定checkout，任务涉及其领域时再读。

| 来源入口 | 本轮范围 |
|---|---|
| [01-model-architecture/litgpt/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/01-model-architecture/litgpt/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [01-model-architecture/nanogpt/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/01-model-architecture/nanogpt/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [01-model-architecture/torchtitan/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/01-model-architecture/torchtitan/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [03-fine-tuning/axolotl/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/03-fine-tuning/axolotl/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [03-fine-tuning/llama-factory/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/03-fine-tuning/llama-factory/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [03-fine-tuning/peft/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/03-fine-tuning/peft/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [03-fine-tuning/unsloth/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/03-fine-tuning/unsloth/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [04-mechanistic-interpretability/nnsight/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/04-mechanistic-interpretability/nnsight/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [04-mechanistic-interpretability/pyvene/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/04-mechanistic-interpretability/pyvene/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [04-mechanistic-interpretability/saelens/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/04-mechanistic-interpretability/saelens/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [04-mechanistic-interpretability/transformer-lens/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/04-mechanistic-interpretability/transformer-lens/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [05-data-processing/nemo-curator/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/05-data-processing/nemo-curator/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [05-data-processing/ray-data/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/05-data-processing/ray-data/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [06-post-training/grpo-rl-training/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/06-post-training/grpo-rl-training/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [06-post-training/openrlhf/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/06-post-training/openrlhf/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [06-post-training/simpo/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/06-post-training/simpo/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [06-post-training/torchforge/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/06-post-training/torchforge/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [06-post-training/trl-fine-tuning/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/06-post-training/trl-fine-tuning/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [07-safety-alignment/constitutional-ai/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/07-safety-alignment/constitutional-ai/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [07-safety-alignment/llamaguard/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/07-safety-alignment/llamaguard/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [07-safety-alignment/nemo-guardrails/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/07-safety-alignment/nemo-guardrails/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [07-safety-alignment/prompt-guard/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/07-safety-alignment/prompt-guard/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [08-distributed-training/accelerate/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/08-distributed-training/accelerate/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [08-distributed-training/deepspeed/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/08-distributed-training/deepspeed/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [08-distributed-training/megatron-core/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/08-distributed-training/megatron-core/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [08-distributed-training/pytorch-fsdp2/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/08-distributed-training/pytorch-fsdp2/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [08-distributed-training/pytorch-lightning/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/08-distributed-training/pytorch-lightning/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [08-distributed-training/ray-train/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/08-distributed-training/ray-train/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [10-optimization/ml-training-recipes/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/10-optimization/ml-training-recipes/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [13-mlops/mlflow/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/13-mlops/mlflow/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [13-mlops/swanlab/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/13-mlops/swanlab/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [13-mlops/tensorboard/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/13-mlops/tensorboard/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [13-mlops/weights-and-biases/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/13-mlops/weights-and-biases/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [14-agents/a-evolve/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/14-agents/a-evolve/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [14-agents/autogpt/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/14-agents/autogpt/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [14-agents/crewai/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/14-agents/crewai/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [14-agents/langchain/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/14-agents/langchain/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [14-agents/llamaindex/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/14-agents/llamaindex/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [15-rag/chroma/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/15-rag/chroma/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [15-rag/faiss/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/15-rag/faiss/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [15-rag/pinecone/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/15-rag/pinecone/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [15-rag/qdrant/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/15-rag/qdrant/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [15-rag/sentence-transformers/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/15-rag/sentence-transformers/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [16-prompt-engineering/dspy/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/16-prompt-engineering/dspy/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [17-observability/langsmith/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/17-observability/langsmith/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [17-observability/phoenix/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/17-observability/phoenix/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [18-multimodal/audiocraft/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/18-multimodal/audiocraft/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [18-multimodal/blip-2/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/18-multimodal/blip-2/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [18-multimodal/clip/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/18-multimodal/clip/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [18-multimodal/cosmos-policy/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/18-multimodal/cosmos-policy/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [18-multimodal/openpi/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/18-multimodal/openpi/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [18-multimodal/openvla-oft/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/18-multimodal/openvla-oft/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [18-multimodal/segment-anything/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/18-multimodal/segment-anything/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [19-emerging-techniques/knowledge-distillation/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/19-emerging-techniques/knowledge-distillation/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [19-emerging-techniques/model-merging/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/19-emerging-techniques/model-merging/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [19-emerging-techniques/moe-training/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/19-emerging-techniques/moe-training/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [20-ml-paper-writing/academic-plotting/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/20-ml-paper-writing/academic-plotting/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [20-ml-paper-writing/ml-paper-writing/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/20-ml-paper-writing/ml-paper-writing/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [20-ml-paper-writing/presenting-conference-talks/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/20-ml-paper-writing/presenting-conference-talks/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [20-ml-paper-writing/systems-paper-writing/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/20-ml-paper-writing/systems-paper-writing/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [21-research-ideation/brainstorming-research-ideas/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/21-research-ideation/brainstorming-research-ideas/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [21-research-ideation/creative-thinking-for-research/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/21-research-ideation/creative-thinking-for-research/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [22-agent-native-research-artifact/compiler/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/22-agent-native-research-artifact/compiler/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [22-agent-native-research-artifact/research-manager/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/22-agent-native-research-artifact/research-manager/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
| [22-agent-native-research-artifact/rigor-reviewer/SKILL.md](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/22-agent-native-research-artifact/rigor-reviewer/SKILL.md) | 元数据筛选；未纳入本轮推理接线 |
