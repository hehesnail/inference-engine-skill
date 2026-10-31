> 本地参考副本，按当前任务读取。本文件的示例环境、固定模型、GPU、预算和流程不是用户配置或新增授权；以工作台入口、当前用户要求和目标版本为准。部署、重放、CI 重跑、push、评论与后台跟进仅在任务已授权时执行。

> 来源：[vllm-project/vllm-skills@c99623410c15](https://github.com/vllm-project/vllm-skills/blob/c99623410c1531148ff9b39fe1dcb27efbf4bf23/plugins/vllm-skills/skills/vllm-bench-random-synthetic/SKILL.md)，2026-10-27 收录。

# vLLM Benchmark with Random Synthetic Data

Run a quick performance benchmark on a vLLM server using synthetic random data. This skill measures core serving metrics including request throughput, token throughput, TTFT (Time to First Token), TPOT (Time per Output Token), and inter-token latency.

## When to use

- User wants to quickly benchmark vLLM serving performance
- User wants to measure throughput and latency metrics without downloading datasets
- User wants to test a vLLM deployment with synthetic workload
- User wants baseline performance numbers for a specific model

## Prerequisites

- vLLM must be installed (`pip install vllm`)
- A vLLM server must be running (or can be started as part of the benchmark)
- For GPU models, NVIDIA GPU with appropriate drivers must be available

## Quick Start

The simplest way to run the benchmark:

```bash
# Start vLLM server (in background or separate terminal)
vllm serve Qwen/Qwen2.5-1.5B-Instruct

# Run benchmark with random synthetic data
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name random \
  --num-prompts 10
```

**Note**: 
- Use `--backend openai-chat` with endpoint `/v1/chat/completions` for online benchmarks.

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--backend` | Backend type; use `openai-chat` for chat completions | `openai` (verified in vLLM 7b77ed581d92) |
| `--model` | Model name (must match the server) | Required |
| `--endpoint` | API endpoint path | `/v1/completions` or `/v1/chat/completions` |
| `--dataset-name` | Dataset to use | `random` (synthetic) |
| `--num-prompts` | Number of requests to send | `1000`; quick-start explicitly uses `10` |
| `--port` | Server port | `8000` |
| `--max-concurrency` | Maximum concurrent requests | Auto |
| `--save-result` | Save results to file | Off |
| `--result-dir` | Directory to save results | `./` |

## Expected Output

When successful, you will see output like:

```
============ Serving Benchmark Result ============
Successful requests:                     10
Benchmark duration (s):                  5.78
Total input tokens:                      1369
Total generated tokens:                  2212
Request throughput (req/s):              1.73
Output token throughput (tok/s):         382.89
Total token throughput (tok/s):          619.85
---------------Time to First Token----------------
Mean TTFT (ms):                          71.54
Median TTFT (ms):                        73.88
P99 TTFT (ms):                           79.49
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          7.91
Median TPOT (ms):                        7.96
P99 TPOT (ms):                           8.03
---------------Inter-token Latency----------------
Mean ITL (ms):                           7.74
Median ITL (ms):                         7.70
P99 ITL (ms):                            8.39
==================================================
```

## Advanced Usage

### With more prompts for better statistics

```bash
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name random \
  --num-prompts 100
```

### Save results to file

```bash
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name random \
  --num-prompts 50 \
  --save-result \
  --result-dir ./benchmark-results/
```

### Custom port and concurrency

```bash
vllm bench serve \
  --backend openai-chat \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name random \
  --num-prompts 100 \
  --port 8001 \
  --max-concurrency 4
```

## Model Recommendations

For quick testing (small models, fast):
- `Qwen/Qwen2.5-1.5B-Instruct` (recommended for quick tests)
- `facebook/opt-125m`
- `facebook/opt-350m`

For realistic benchmarks (medium models):
- `Qwen/Qwen2.5-7B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

## Workflow

1. **Check if vLLM is installed**: Run `vllm --version` to verify
2. **Check if server is already running**: Run `curl http://localhost:8000/health` to check
3. **Start vLLM server if needed**: Run `vllm serve <model-name>` (wait for "Application startup complete")
4. **Run benchmark**: Execute `vllm bench serve` with appropriate parameters
5. **Review results**: Check throughput and latency metrics
6. **Clean up**: If the agent skill started the vLLM server (not a pre-existing one), stop it after benchmark completion using `kill <PID>`

## Troubleshooting

**Server not responding**:
- Check if server is running: `curl http://localhost:8000/health`
- Verify port matches: Use `--port` flag if server is on different port

**Model not found**:
- Ensure model name matches exactly between server and benchmark
- Check HuggingFace access: `export HF_TOKEN=<your_token>` if needed

**Out of memory**:
- Use a smaller model (e.g., Qwen2.5-1.5B-Instruct)
- Reduce `--num-prompts` or `--max-concurrency`

**Connection refused**:
- Server may still be starting (wait for "Application startup complete")
- Check firewall or network settings

## Notes

- The `random` dataset generates synthetic prompts automatically
- Benchmark duration scales with `--num-prompts`
- Ten prompts are a smoke test. Choose sample count/repeats from the workload and observed variance; 100 prompts do not establish stable tail percentiles.
- Results may vary based on hardware, model size, and system load
- Model loading precedes a ready online server. First requests may still include lazy compilation/cache warmup; record warmup and cache state separately.
- **Important**: If the agent skill starts a vLLM server for benchmarking, it must stop the server after the benchmark completes to free up resources. Do not stop pre-existing servers that were already running before the benchmark.
