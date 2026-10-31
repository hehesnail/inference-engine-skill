> 本地核对：profile应针对被测环境，不能强制换成latest镜像。decode step duration通常是该batch各请求的下一token间隔；除以batch size得到的是吞吐摊销口径，不是单请求TPOT。kernel名称和默认graph行为按目标版本确认。

> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[intel/gpu-ai-skills@d0f47fcf2625](https://github.com/intel/gpu-ai-skills/blob/d0f47fcf262560b06fea1684c70d80c877e04854/plugins/intel-gpu-ai-skills/skills/vllm-xpu-profile/SKILL.md)，2026-10-31 核对。

# vllm-xpu-profile

Profile a vLLM-XPU server with `torch.profiler` to see scheduler,
KV manager, attention backend, and batching alongside XPU op
timeline. For pure-PyTorch traces use **torch-xpu-profile**; for
SYCL kernel level use **xpu-profile-unitrace**.

## Modes

| Mode | When |
|---|---|
| **A** — running server + HTTP `/start_profile` ... `/stop_profile` | Real-traffic capture; see scheduler / KV manager / batching behaviour. |
| **B** — `vllm bench latency / throughput --profile` | Offline; no network endpoint needed. |

## Mode A — server + HTTP bracket

Profile the actual target image/source revision with version-supported profiler flags. Preserve the workload and configuration. An upstream image can be a separately labeled comparison, but replacing the target with `latest` changes the system under investigation.

```sh
docker run -d --name vllm-xpu-prof \
    --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path:ro \
    --group-add "$(getent group render | cut -d: -f3)" \
    --ipc=host \
    -e ZE_AFFINITY_MASK=0 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
   -e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY \
   -e http_proxy -e https_proxy -e no_proxy \
   -e HF_TOKEN \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -v "$PWD/traces:/work/traces" \
    -p 8000:8000 \
   vllm/vllm-openai-xpu:latest \
   Qwen/Qwen2.5-1.5B-Instruct \
        --dtype bfloat16 --max-model-len 4096 \
        --profiler-config.profiler=torch \
        --profiler-config.torch_profiler_dir=/work/traces
```

Wait for `Application startup complete`, then bracket your window:

```sh
# Warmup (don't profile this) — kernel cache hot
for i in 1 2 3; do
    curl -s http://localhost:8000/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct",
             "messages":[{"role":"user","content":"hi"}],"max_tokens":32}' >/dev/null
done

curl -X POST http://localhost:8000/start_profile

# Workload to characterise
for i in $(seq 1 8); do
    curl -s http://localhost:8000/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct",
             "messages":[{"role":"user","content":"Write a short paragraph."}],
             "max_tokens":128}' &
done; wait

curl -X POST http://localhost:8000/stop_profile
ls traces/
```

`Qwen/Qwen2.5-1.5B-Instruct` is public, so `HF_TOKEN` is optional;
it is forwarded for gated/private replacements and anonymous Hub rate
limits. `-e HF_TOKEN` (no `=value`) passes through the host's already-exported
variable without putting the token in the `docker` process argv, where
`ps`/`/proc` on a shared host could expose it. The proxy lines are
required on hosts whose containers cannot reach the Hub directly.
Docker ignores unset proxy variable names.

Cleanup after the trace has flushed:

```sh
docker stop vllm-xpu-prof && docker rm vllm-xpu-prof
```

Three artifacts written per profile (allow a few seconds for flush):

- `<timestamp>-rank-<N>.<timestamp>.pt.trace.json.gz` — per-worker
  Chrome trace. Drag-and-drop into <https://ui.perfetto.dev>;
  Perfetto handles `.json.gz` directly.
- `<container>_<N>.async_llm.<timestamp>.pt.trace.json.gz` — async
  engine trace (smaller).
- `profiler_out_<N>.txt` — **read this first**, sort-by-XPU-time
  table of top kernels. Most "what's slow" questions are answered
  by its top 5 rows:

  ```
  Name                     Self XPU %  XPU total   # of Calls
  gemm_kernel              28.50%      26.232ms    4753
  aten::mm                 21.79%      20.061ms    3577
  _C::fused_add_rms_norm   14.11%      12.988ms    2352
  _vllm_fa2_C::varlen_fwd   6.70%      17.979ms    1176
  _C::rotary_embedding      6.70%       6.172ms    1176
  Memcpy H2D                2.11%       1.945ms     412
  ```

## Mode B — offline bench with `--profile`

```sh
docker run --rm \
   --entrypoint vllm \
    --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path:ro \
    --group-add "$(getent group render | cut -d: -f3)" \
    --ipc=host \
    -e ZE_AFFINITY_MASK=0 \
   -e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY \
   -e http_proxy -e https_proxy -e no_proxy \
   -e HF_TOKEN \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -v "$PWD/traces:/work/traces" \
   vllm/vllm-openai-xpu:latest \
   bench latency \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --dtype bfloat16 \
        --num-iters-warmup 2 --num-iters 5 \
        --profile \
        --profiler-config.profiler=torch \
        --profiler-config.torch_profiler_dir=/work/traces
```

## What to look for

1. **Step boundary markers** — `EngineCore.step` spans. A decode step may advance many requests together; step duration is not divided by batch size to obtain per-request TPOT. That division is a throughput amortization. Trace each request/token interval and account for scheduler gaps.
2. **Attention path** — confirm kernel name matches your
   `--attention-backend`. If you passed `TRITON_ATTN` but trace
   shows `flash_attention_xpu`, confirm the flag reached the
   engine:
   ```sh
   docker logs <name> 2>&1 | grep -E "attention_backend|Using.*backend"
   ```
3. **Gaps between steps** — wide white bands at boundaries mean
   scheduler / KV manager is on the critical path. Expected to be
   tight on v0.31.0+, where XPU graphs capture by default; wide bands
   there are a real finding. They are normal under `--enforce-eager`,
   so check whether the server was launched with it before chasing
   them. Profile the configuration you actually serve — adding eager
   just to get a cleaner trace changes the thing you are measuring,
   though it is a legitimate second run when you need per-kernel
   granularity that graph capture collapses.
4. **Quant kernel names** — for `--quantization fp8` expect
   `fp8_gemm_w8a16`; for `awq` expect `int4_gemm_w4a8`. Seeing
   `int4_gemm_w4a16` with AWQ -> silent fall-through (vLLM #38064).
5. **Per-rank spans** — multi-GPU `-tp N` shows N timelines. Skew
   between ranks means a slow rank is gating throughput.

## Common errors

- `torch_profiler_dir is only applicable when profiler is set to 'torch'` -> add `--profiler-config.profiler=torch`.
- `/start_profile` 404 -> server launched without profiler-config flags; restart.
- Empty trace -> too few requests inside the bracket; send 5–10+.
- Trace doesn't appear after `/stop_profile` -> flush is async; wait a few seconds, or check server logs.

## Env vars

| Variable | Purpose |
|---|---|
| `ZE_AFFINITY_MASK` | Pin to one XPU. |
| `PYTORCH_ENABLE_XPU_FALLBACK=0` | Error on CPU-fallback so it can't hide. |
| `KINETO_LOG_LEVEL=INFO` | Diagnose incomplete trace export. |

(vLLM profiler config is via CLI flags `--profiler-config.*`, not env vars.)

## What this skill does NOT cover

- Pure-PyTorch profiling (no server) -> **torch-xpu-profile**.
- SYCL-kernel events / hardware metrics -> **xpu-profile-unitrace**.
- Fixing a hot kernel — out of scope.
- Variant images may have different profiler flags and internals. Inspect their installed source and preserve their identity; use an upstream comparison only when relevant to the task.

## References

- vLLM profiling guide: <https://docs.vllm.ai/en/latest/contributing/profiling/>
- Perfetto trace viewer: <https://ui.perfetto.dev>
