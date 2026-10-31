> 本地参考副本，按当前任务读取。本文件的示例环境、固定模型、GPU、预算和流程不是用户配置或新增授权；以工作台入口、当前用户要求和目标版本为准。部署、重放、CI 重跑、push、评论与后台跟进仅在任务已授权时执行。

> 来源：[vllm-project/vllm-skills@c99623410c15](https://github.com/vllm-project/vllm-skills/blob/c99623410c1531148ff9b39fe1dcb27efbf4bf23/plugins/vllm-skills/skills/vllm-prefix-cache-bench/SKILL.md)，2026-10-27 收录。

# vLLM Prefix Caching Benchmark

Benchmark the efficiency of vLLM's automatic prefix caching (APC) feature. The offline script `benchmarks/benchmark_prefix_caching.py` runs directly against the vLLM engine (no server required). For online/serving tests, use `vllm bench serve` with the `prefix_repetition` dataset.

## When to use

- User wants to measure the performance impact of prefix caching for repeated or partially-shared prompts.
- User wants to compare throughput/latency with and without `--enable-prefix-caching`.
- User wants to test prefix caching using a fixed synthetic prompt, a real dataset (e.g. ShareGPT), or a synthetic prefix/suffix repetition pattern.

## Option 1 (default). Fixed Prompt with Prefix Caching

Samples synthetic prompt(s), repeats them in one engine generation batch, and prints generation wall time. It does not report hit rate or prove that all repetitions hit the cache. Record scheduler/batching and cache initial state, and collect cache metrics separately when hit efficiency is the question. No dataset download required.

```bash
python3 benchmarks/benchmark_prefix_caching.py \
  --model Qwen/Qwen3-8B \
  --enable-prefix-caching \
  --num-prompts 1 \
  --repeat-count 100 \
  --input-length-range 128:256
```

To compare against the baseline without caching:

```bash
python3 benchmarks/benchmark_prefix_caching.py \
  --model Qwen/Qwen3-8B \
  --no-enable-prefix-caching \
  --num-prompts 1 \
  --repeat-count 100 \
  --input-length-range 128:256
```

## Option 2. ShareGPT Dataset with Prefix Caching

Uses real-world conversational data from ShareGPT to evaluate prefix caching with naturally occurring prompt sharing.

First, download the dataset:

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

Then run the benchmark:

```bash
python3 benchmarks/benchmark_prefix_caching.py \
  --model Qwen/Qwen3-8B \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --enable-prefix-caching \
  --num-prompts 20 \
  --repeat-count 5 \
  --input-length-range 128:256
```

## Option 3. Prefix Repetition Dataset (Online)

Uses `vllm bench serve` with the synthetic `prefix_repetition` dataset to test caching via the serving API. This requires a running vLLM server.

First, start the server:

```bash
vllm serve Qwen/Qwen3-8B
```

Then run the benchmark:

```bash
vllm bench serve \
  --backend openai \
  --model Qwen/Qwen3-8B \
  --dataset-name prefix_repetition \
  --num-prompts 100 \
  --prefix-repetition-prefix-len 512 \
  --prefix-repetition-suffix-len 128 \
  --prefix-repetition-num-prefixes 5 \
  --prefix-repetition-output-len 128
```

Key parameters for `prefix_repetition`:

| Parameter | Description |
|---|---|
| `--prefix-repetition-prefix-len` | Number of tokens in the shared prefix portion |
| `--prefix-repetition-suffix-len` | Number of tokens in the unique suffix portion |
| `--prefix-repetition-num-prefixes` | Number of distinct prefixes to cycle through |
| `--prefix-repetition-output-len` | Number of output tokens to generate per request |

## Notes

- Run all commands from the root of the vLLM repository (`cd vllm`).
- Select the task model and workload. The shown checkpoint, lengths and repeat counts are examples, not a fixed comparison protocol.
- `--repeat-count` in Option 1 and 2 repeats sampled prompts in a single submitted list; actual reuse depends on scheduling, block completion and eviction.
- `--input-length-range` accepts a `min:max` token range, e.g. `128:256`.
- For multi-GPU setups, add `--tensor-parallel-size <N>`.
- To test different hash algorithms for prefix caching internals, use `--prefix-caching-hash-algo xxhash` (requires `pip install xxhash`).

## Arguments for `benchmark_prefix_caching.py`

| Argument | Required | Description |
|---|---|---|
| `--model` | Yes | Model name or path (HuggingFace ID or local path) |
| `--num-prompts` | Yes | Number of prompts to process |
| `--input-length-range` | Yes | Token length range for inputs, e.g. `128:256` |
| `--repeat-count` | No | Number of times each prompt is repeated (default: 1) |
| `--dataset-path` | No | Path to a dataset file (e.g. ShareGPT JSON). Omit for synthetic fixed-prompt mode |
| `--prefix-len` | No | Fixed prefix token length to prepend to every prompt |
| `--output-len` | No | Number of output tokens to generate per request |
| `--sort` | No | Historical option: at vLLM 7b77ed581d92 it raises TypeError because Request is a dataclass but the sort key reads x[1]. Omit unless fixed in the target checkout. |
| `--enable-prefix-caching` / `--no-enable-prefix-caching` | No | Toggle APC (recommended: enable to test caching) |
| `--prefix-caching-hash-algo` | No | Hash algorithm: `sha256`, `sha256_cbor`, `xxhash`, `xxhash_cbor` |
| `--tensor-parallel-size` | No | Number of GPUs for tensor parallelism |
| `--disable-detokenize` | No | Skip detokenization to reduce overhead |

## Troubleshooting

- If `python3 benchmarks/*.py` reports file not found, locate your local vLLM repository first and run the command from that repo root.
- If you do not have the repository yet, clone it and continue:

```bash
git clone https://github.com/vllm-project/vllm
cd vllm
```

- If HuggingFace model download fails due to access restrictions, set your token: `export HF_TOKEN=<your_token>` or pass `--hf-token <your_token>`.
- If `xxhash` or `cbor2` is not installed and you use those hash algorithms, install them first: `pip install xxhash cbor2`.
