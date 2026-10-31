# 本地专项资料目录

这里的正文和脚本已随包保存。按当前问题打开一个或少数条目；无需逐一读完。`GUIDE.md` 是资料入口，不注册成额外的全局 skills。

通用离线工具的稳定调用方式见 [tools.md](tools.md)；大语料位置见 [server-data.md](server-data.md)。

## vLLM 部署与服务 benchmark

| 专项 | 何时读取 |
|---|---|
| [vllm-bench-random-synthetic](../library/vllm-skills/plugins/vllm-skills/skills/vllm-bench-random-synthetic/GUIDE.md) | Run vLLM performance benchmark using synthetic random data to measure throughput, TTFT (Time to First Token), TPOT (Time per Output Token), and other key performance metrics |
| [vllm-bench-serve](../library/vllm-skills/plugins/vllm-skills/skills/vllm-bench-serve/GUIDE.md) | Benchmark vLLM or OpenAI-compatible serving endpoints using vllm bench serve |
| [vllm-deploy-docker](../library/vllm-skills/plugins/vllm-skills/skills/vllm-deploy-docker/GUIDE.md) | Deploy vLLM using Docker (pre-built images or build-from-source) with NVIDIA GPU support and run the OpenAI-compatible server. |
| [vllm-deploy-k8s](../library/vllm-skills/plugins/vllm-skills/skills/vllm-deploy-k8s/GUIDE.md) | Deploy vLLM to Kubernetes (K8s) with GPU support, health probes, and OpenAI-compatible API endpoint |
| [vllm-deploy-simple](../library/vllm-skills/plugins/vllm-skills/skills/vllm-deploy-simple/GUIDE.md) | Quick install and deploy vLLM, start serving with a simple LLM, and test OpenAI API. |
| [vllm-prefix-cache-bench](../library/vllm-skills/plugins/vllm-skills/skills/vllm-prefix-cache-bench/GUIDE.md) | This is a skill for benchmarking the efficiency of automatic prefix caching in vLLM using fixed prompts, real-world datasets, or synthetic prefix/suffix patterns |

## vLLM 开发

| 专项 | 何时读取 |
|---|---|
| [check-api-compat](../library/vllm/.agents/skills/check-api-compat/GUIDE.md) | Determine the correct behavior of vLLM's OpenAI-compatible or Anthropic-compatible API by checking the official upstream spec, instead of guessing from vLLM's own code |
| [ci-fails-buildkite](../library/vllm/.agents/skills/ci-fails-buildkite/GUIDE.md) | Fetch and diagnose vLLM Buildkite CI failure logs |
| [debug-ima](../library/vllm/.agents/skills/debug-ima/GUIDE.md) | Debug CUDA illegal memory access (IMA) errors in vLLM with CUDA core dumps and cuda-gdb |
| [kernel-microbenchmark](../library/vllm/.agents/skills/kernel-microbenchmark/GUIDE.md) | Build, debug, and interpret vLLM GPU kernel microbenchmarks for CUDA, ROCm/HIP, Triton, and CuteDSL, including CUPTI timing, correctness checks, generated-code inspection, multi-GPU measurements, and SOL sanity checks |
| [model-microbenchmark](../library/vllm/.agents/skills/model-microbenchmark/GUIDE.md) | Time a vLLM model's forward pass on exact batch shapes (decode, prefill, chunked prefill, prefix hit, spec-decode verify, or mixes) with `vllm bench forward`, single-GPU or DP/TP/EP across nodes |
| [model-recipes](../library/vllm/.agents/skills/model-recipes/GUIDE.md) | Find vLLM deployment recipes and configs for LLMs and Hugging Face models by model and hardware |
| [pr-checklist](../library/vllm/.agents/skills/pr-checklist/GUIDE.md) | Prepare a vLLM change for human PR review or re-review, or analyze an open pull request |
| [transformers-upgrade](../library/vllm/.agents/skills/transformers-upgrade/GUIDE.md) | Playbook for changing vLLM's Transformers version bounds |
| [triton-kernel-writing](../library/vllm/.agents/skills/triton-kernel-writing/GUIDE.md) | Write or review Triton kernels for vLLM, with practical guidance for generated-code inspection, launch grids, indexing, specialization, tuning, and representative performance validation |

## SGLang 专项开发与排障

| 专项 | 何时读取 |
|---|---|
| [add-jit-kernel](../library/sglang/.agents/skills/add-jit-kernel/GUIDE.md) | Step-by-step tutorial for adding a new lightweight JIT CUDA kernel to sglang.kernels JIT infrastructure and public operator groups |
| [add-sgl-kernel](../library/sglang/.agents/skills/add-sgl-kernel/GUIDE.md) | Step-by-step tutorial for adding a heavyweight AOT CUDA/C++ kernel to sgl-kernel (including tests & benchmarks) |
| [babysit-pr-to-pass-ci](../library/sglang/.agents/skills/babysit-pr-to-pass-ci/GUIDE.md) | Start and persistently pursue a goal to babysit an SGLang pull request until selected GitHub Actions workflows pass on the latest PR head |
| [ci-test-audit](../library/sglang/.agents/skills/ci-test-audit/GUIDE.md) | Audit the existing test tree and CI configuration for improvements, using a catalog of patterns previously applied in this repo |
| [ci-workflow-guide](../library/sglang/.agents/skills/ci-workflow-guide/GUIDE.md) | Guide to SGLang CI workflow orchestration — stage ordering, fail-fast, gating, partitioning, execution modes, and debugging CI failures |
| [clean-startup-log](../library/sglang/.agents/skills/clean-startup-log/GUIDE.md) | Audit SGLang startup logs, save evidence, and propose cleanup for user review |
| [compute-mamba-ratio](../library/sglang/.agents/skills/compute-mamba-ratio/GUIDE.md) | Compute the optimal --mamba-full-memory-ratio (or --max-mamba-cache-size pin) for a hybrid attention + linear-attention (Mamba / GDN / KDA) model's two serving memory pools, from the workload and serving config |
| [cookbook-add-model](../library/sglang/.agents/skills/cookbook-add-model/GUIDE.md) | Add a new model to the SGLang Cookbook (docs/, Mintlify), config-driven format — instantiate the model-agnostic template into a per-model config (+ benchmarks) JSX under src/snippets/configs/, an MDX page, the docs.json nav entry, NEW-tag hygiene, and the homepage vendor card |
| [cookbook-migrate-model](../library/sglang/.agents/skills/cookbook-migrate-model/GUIDE.md) | Migrate a legacy-template SGLang cookbook page (monolithic per-model generator under docs/src/snippets/autoregressive/) onto the config-driven template (shared _deployment.jsx / _playground.jsx engines + per-model config) |
| [cookbook-review-pr](../library/sglang/.agents/skills/cookbook-review-pr/GUIDE.md) | Review a pull request against the SGLang Cookbook (docs/, Mintlify) contribution checklist — the config-driven format (per-model config + benchmarks JSX consumed by the shared _deployment.jsx / _playground.jsx engines) |
| [debug-cuda-crash](../library/sglang/.agents/skills/debug-cuda-crash/GUIDE.md) | Call this skill when you need to debug CUDA crashes in SGLang using kernel API logging |
| [debug-distributed-hang](../library/sglang/.agents/skills/debug-distributed-hang/GUIDE.md) | Debug hanging issues in SGLang distributed inference (TP/PP/DP/EP) |
| [env-var-conventions](../library/sglang/.agents/skills/env-var-conventions/GUIDE.md) | Conventions for SGLang environment variables — where to define, how to access, how to name, and how to deprecate |
| [generate-profile](../library/sglang/.agents/skills/generate-profile/GUIDE.md) | Generate an e2e profiling trace of an SGLang server run |
| [kernel-organization](../library/sglang/.agents/skills/kernel-organization/GUIDE.md) | Apply the SGLang kernels RFC when adding, moving, splitting, or reviewing kernel APIs, registry metadata, kernel tests, benchmarks, and model-specific implementations |
| [kl-consistency-test](../library/sglang/.agents/skills/kl-consistency-test/GUIDE.md) | Write, calibrate, and debug the prefill-vs-decode logprob (KL) consistency tests in sglang -- the two independent conditions a zero requires (every operator batch-invariant, and the two paths computing the same function), which helper separates them, how to pick a threshold once they hold, and how to localize a divergence to a single operator |
| [large-class-style](../library/sglang/.agents/skills/large-class-style/GUIDE.md) | Code style for SGLang large classes `Scheduler`, `TokenizerManager`, and `ModelRunner`: frozen-code conventions and `__init__` orchestration style |
| [llm-torch-profiler-analysis](../library/sglang/.agents/skills/llm-torch-profiler-analysis/GUIDE.md) | Unified LLM torch-profiler triage skill for `sglang`, `vllm`, `TensorRT-LLM`, and `TokenSpeed` |
| [mechanical-refactor-verify](../library/sglang/.agents/skills/mechanical-refactor-verify/GUIDE.md) | Make mechanical refactoring (file splits, function moves, module extractions, renames) machine-checkable instead of eyeballed |
| [processor-model-parity](../library/sglang/.agents/skills/processor-model-parity/GUIDE.md) | Add or verify a model's chat prompt rendering and tokenization in rust/sglang-processor so it matches SGLang's Python serving path exactly (same prompt text, same token ids) |
| [scripted-runtime-notes](../library/sglang/.agents/skills/scripted-runtime-notes/GUIDE.md) | Requirements for the SGLang scripted runtime, chiefly when to add (vs not add) a harness API |
| [sglang-bisect-ci-regression](../library/sglang/.agents/skills/sglang-bisect-ci-regression/GUIDE.md) | Investigate consistently failing SGLang CI tests by extracting the failure signature from scheduled or rerun workflows, bisecting the passing/failing commit window, checking runner or hardware specificity, and optionally reproducing on a remote GPU host. |
| [sglang-cherrypick](../library/sglang/.agents/skills/sglang-cherrypick/GUIDE.md) | Trigger the bot-cherry-pick workflow for a batch of merged PRs onto a release branch and monitor each run to completion |
| [sglang-prod-incident-triage](../library/sglang/.agents/skills/sglang-prod-incident-triage/GUIDE.md) | Replay-first debug flow for SGLang serving problems |
| [sglang-runtime-context](../library/sglang/.agents/skills/sglang-runtime-context/GUIDE.md) | How SGLang's runtime configuration and process-global state are organized (RuntimeContext tiers, publish + namespace config bags, the pristine ServerArgs seed, override entry points, resource/stream/buffer leases, per-forward flags), the CI guardrails that enforce the design, and the idioms for developing and testing against it |
| [speculative-naming](../library/sglang/.agents/skills/speculative-naming/GUIDE.md) | Naming conventions for SGLang speculative decoding identifiers |
| [write-sglang-test](../library/sglang/.agents/skills/write-sglang-test/GUIDE.md) | Guide for writing SGLang CI/UT tests |

## BBuf 分析工具与知识库

| 专项 | 何时读取 |
|---|---|
| [llm-pipeline-analysis](../library/bbuf/skills/llm-pipeline-analysis/GUIDE.md) | Inspect LLM torch profiler traces at forward-pass, layer, and kernel level |
| [llm-serving-auto-benchmark](../library/bbuf/skills/llm-serving-auto-benchmark/GUIDE.md) | Framework-independent LLM serving benchmark skill for comparing SGLang, vLLM, TensorRT-LLM, TokenSpeed, or another serving framework |
| [llm-serving-capacity-planner](../library/bbuf/skills/llm-serving-capacity-planner/GUIDE.md) | Parse SGLang/vLLM startup logs to explain GPU memory use and request capacity |
| [llm-torch-profiler-analysis](../library/bbuf/skills/llm-torch-profiler-analysis/GUIDE.md) | Analyze existing Torch Profiler Chrome traces or capture from SGLang, vLLM, TensorRT-LLM and TokenSpeed servers |
| [model-compute-simulation](../library/bbuf/skills/model-compute-simulation/GUIDE.md) | Build an operator-level compute template for an LLM and estimate FLOPs/MFU for a serving shape |
| [model-pr-history-knowledge](../library/bbuf/model-pr-optimization-history/GUIDE.md) | Use when an SGLang, vLLM, TensorRT-LLM, or TokenSpeed serving/model optimization task needs prior model-family PR evidence |
| [sglang-humanize-review](../library/bbuf/skills/sglang-humanize-review/GUIDE.md) | Perform SGLang code review in the style of human maintainers by consulting the public PR review episode corpus from project start through 2026-07-27, including inline review threads, top-level PR comments, review submissions, original multilingual text, and multi-round discussions |
| [sglang-model-day0-support](../library/bbuf/skills/model-optimization/sglang-model-day0-support/GUIDE.md) | Build or audit an evidence-driven SGLang Day-0 support program for a new LLM, VLM, MoE, hybrid-attention, or speculative-decoding model |
| [sglang-prod-incident-triage](../library/bbuf/skills/sglang-prod-incident-triage/GUIDE.md) | Replay-first debug flow for SGLang serving problems |
| [torch-profiler-layer-track](../library/bbuf/skills/torch-profiler-layer-track/GUIDE.md) | Add verified layer-number guides and compact GPU display lanes to an existing Torch Profiler Chrome JSON trace |

## 扩展专项

新增完整工程与SGLang diffusion内容见[逐工程目录](repository-guides.md)。模型权重和依赖环境留在工作区指定位置。

## 本轮补强的推理工程专题

| 专项 | 何时读取 |
|---|---|
| [Host/GPU 归因](host-device-analysis.md) · [TRT 完整专题资料](../library/trtllm/host/GUIDE.md) | CPU launch、输入准备、step 预算、通信尾部与图模式 |
| [NCU 指导](nsight-compute.md) · [完整指南/脚本](../library/ncu-report/GUIDE.md) | kernel counters、PC stall、roofline、报告解析 |
| [kernel 接线](kernel-engine-integration.md) · [Compute Sanitizer](../library/nvidia/compute-sanitizer/GUIDE.md) | 原位/stride、dispatch/fallback、custom op 与 CUDA 正确性 |
| [Graph 与同步](cuda-graphs-and-sync.md) | buffer 生命周期、异步消费、compile/fusion |
| [优化候选](optimization-workflow.md) | 原始基线/最好版本/候选、实际生效配置和有效证据 |
| [fork 维护](fork-maintenance.md) | 升级、rebase、组合回归 |
| [平台与传输](platforms-and-transport.md) | Hopper/Blackwell/ROCm/XPU、TP/EP、PD 与跨引擎对照 |

逐仓核对、吸收位置和未采用部分见 [上游评估](upstream-evaluation.md)。

原有官方vLLM15项的逐项核对见[审计](upstream-vllm-audit.md)。新增专项、48篇技术页及工具闭环见[工程目录](repository-guides.md)和[KernelWiki](kernel-knowledge.md)。

## 语义核对后补齐的实现专题

| 问题 | 按需读取 |
|---|---|
| attention布局、causal对齐、RoPE与GQA | [attention契约](attention-backend-contracts.md) |
| AWQ/GPTQ/HQQ/bnb/GGUF与稀疏执行 | [量化契约](quantization-engine-contracts.md) |
| speculative接受、残差与回滚 | [推测解码](speculative-decoding-contracts.md) |
| Mamba/RWKV状态及rollout权重更新 | [状态生命周期](stateful-inference.md) |
| 模板、tokenizer、grammar与retry | [输入输出契约](tokenizer-and-structured-output.md) |
| VLM、Whisper与diffusion | [多模态接线](multimodal-inference.md) |
