# DP/EP Serving Reference

Use this reference when the user asks how to start a DP model, run PyTorch DP/EP serving, use `lmdeploy serve proxy`, configure `--dp` or `--ep`, set `LMDEPLOY_DP_MASTER_*`, tune DeepEP capacity envs, or smoke-test DP/EP through `torchrun`.

## Serve Startup Shape

For PyTorch backend DP/EP serving, start the proxy and API server as separate long-lived processes. Clients normally call the proxy port. The device lists in the examples illustrate a placement. Substitute the devices allocated to the job before launching.

```bash
export PROXY_HOST=127.0.0.1  # single-node example
export PROXY_PORT=8100
export PROXY_URL=http://${PROXY_HOST}:${PROXY_PORT}

export LMDEPLOY_DP_MASTER_ADDR=${PROXY_HOST}
export LMDEPLOY_DP_MASTER_PORT=29555

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MODEL_PATH=/path/to/model
export DP=2
export TP=1
export EP=2
export MAX_BATCH_SIZE=512
export MAX_PREFILL_TOKEN_NUM=8192
export CACHE_MAX_ENTRY_COUNT=0.8

# Useful for DeepEP MoE capacity planning.
export DEEPEP_MAX_TOKENS_PER_RANK=${MAX_BATCH_SIZE}
export DEEPEP_MAX_BATCH_SIZE=${MAX_BATCH_SIZE}
```

Start the proxy from the intended LMDeploy installation. For a clean benchmark, select a run-owned installation if the shared one contains registrations from another job. The proxy can load `proxy_config.json` at import time.

```bash
lmdeploy serve proxy \
  --server-name ${PROXY_HOST} \
  --server-port ${PROXY_PORT} \
  --routing-strategy min_expected_latency \
  --serving-strategy Hybrid \
  --log-level INFO
```

API server process:

```bash
lmdeploy serve api_server ${MODEL_PATH} \
  --backend pytorch \
  --dp ${DP} \
  --tp ${TP} \
  --ep ${EP} \
  --max-prefill-token-num ${MAX_PREFILL_TOKEN_NUM} \
  --max-batch-size ${MAX_BATCH_SIZE} \
  --cache-max-entry-count ${CACHE_MAX_ENTRY_COUNT} \
  --proxy-url ${PROXY_URL} \
  --nnodes 1 \
  --node-rank 0
```

## Optional Ray Pre-Start

`ray start` is not required for the normal single-node launch path. Use it only when you need an explicit Ray head, a fresh cluster, or custom Ray resource settings.

If workers appear outside the selected GPUs, check which Ray head they joined and which resources it advertises. Reuse it if its placement matches this run. For a fresh head, use an isolated job environment. Stop an existing cluster only when this run owns it.

```bash
ray start --head --num-cpus ${EP} --port 0 --object-store-memory 34359738368
```

## PyTorch Profiler For DP/EP GPU Timeline

For an existing worker trace, start with the request time, worker rank, and capture window. For new LMDeploy DP/EP captures, use the built-in PyTorch profiler by default. Use Nsight Systems when the task calls for an Nsight capture, and honor any tool or collection limits for that run.

Set the profiler environment before starting `lmdeploy serve api_server`, and make sure the envs are propagated into Ray workers:

```bash
export LMDEPLOY_PROFILE_CPU=1
export LMDEPLOY_PROFILE_CUDA=1
export LMDEPLOY_PROFILE_DELAY=60
export LMDEPLOY_PROFILE_DURATION=30
export LMDEPLOY_PROFILE_OUT_PREFIX=/path/to/profile_dir/lmdeploy_profile_
```

For `dp > 1`, `LMDEPLOY_PROFILE_DURATION` must be positive. LMDeploy disables the profiler for DP when `LMDEPLOY_PROFILE_DURATION <= 0` because indefinite profiling does not dump reliably across DP worker shutdown.

The profile output is one Chrome trace JSON per rank:

```text
/path/to/profile_dir/lmdeploy_profile_0.json
/path/to/profile_dir/lmdeploy_profile_1.json
...
```

Open these files in Chrome tracing or Perfetto. Useful ranges already recorded by the engine include `inputs_H2D`, `outputs_D2H`, `forward_cudagraph`, `forward_eager`, `prepare_inputs_for_generation`, `build_step_context`, scheduler ranges, sampling ranges, and `StepInputs` update ranges.

Recommended workflow for serving:

1. Start proxy normally.
2. Export the profiler envs before starting the API server.
3. Start API server and wait until it can answer a real completion, not only `/v1/models`.
4. Start the benchmark so steady-state traffic overlaps the `[delay, delay + duration]` profiler window.
5. After `LMDEPLOY_PROFILE_DURATION` elapses, each rank dumps its trace automatically; stop the serve only after the files appear or the profiler logs its dump.

Keep the duration short, such as 20-60 seconds. PyTorch profiler traces can be large and profiling itself hurts throughput, so use separate no-profile benchmark runs for final performance numbers.

### DP/EP Profiler Pitfalls

- `LMDEPLOY_PROFILE_DELAY` is measured from each `BaseModelAgent.start()` call inside the worker, not from the outer `lmdeploy serve api_server` process launch. Large models may spend minutes loading weights before the profiler delay starts.
- A healthy `/v1/models` response is not enough to prove the trace window contains model work; send at least one real completion during the profiler window.
- PyTorch traces can contain both CPU `record_function` ranges and CUDA timeline ranges with the same names. When counting custom ranges, separate CPU events from CUDA events or the event count may look doubled.
- Trace size scales quickly with duration and rank count. For example, a 90-second four-rank DP/EP run on a 397B FP8 model produced about 356 MB per rank.
- Ray metrics/exporter warnings are often noisy during short profiler runs. If requests succeed and Chrome traces dump, these warnings are usually non-fatal.
- PyTorch profiler warnings such as `External init callback must run in same thread` can appear in Ray workers; treat them as suspicious only if trace files are missing or incomplete.

### DP/EP Shutdown Cleanup

Stopping the foreground command may not stop all DP/EP worker processes. After a profiled run:

1. Check `nvidia-smi` for unexpected memory held on the GPUs used by the run.
2. Inspect remaining `lmdeploy`, `raylet`, and `RayWorkerWrapper` processes.
3. Terminate only the processes launched for the current run, then re-check GPU memory.

Avoid broad cleanup of a shared Ray or Docker environment unless the run owns it or the user explicitly asks for a full reset.

## Notes

- `LMDEPLOY_DP_MASTER_ADDR` and `LMDEPLOY_DP_MASTER_PORT` must be visible to all DP ranks/processes. For multi-node, use a reachable host/IP instead of a bind-only address.
- In DP mode with `--proxy-url`, leaving `--server-port` unset lets LMDeploy
  auto-select per-DP API-server ports. If `--server-port P` is explicitly set,
  LMDeploy respects it and validates the contiguous range
  `P..P+dp_per_node-1`.
- Choose `DP`, `TP`, and `EP` according to the model and visible GPU layout. `DP` is data parallel replica count; `EP` is expert parallelism for MoE; `TP` is tensor parallelism.
- Keep `--max-batch-size`, `--max-prefill-token-num`, `--cache-max-entry-count`, `DEEPEP_MAX_TOKENS_PER_RANK`, and `DEEPEP_MAX_BATCH_SIZE` aligned with the workload and GPU memory.
- Optional profiling envs include `LMDEPLOY_PROFILE_CPU`, `LMDEPLOY_PROFILE_CUDA`, `LMDEPLOY_PROFILE_OUT_PREFIX`, `LMDEPLOY_PROFILE_DELAY`, and `LMDEPLOY_PROFILE_DURATION`.

## Pipeline DP/EP Smoke Test Under torchrun

For a pipeline smoke test that exercises DP ranks directly, launch one Python process per rank and set `dp_rank` in `PytorchEngineConfig`.

```python
import os

import torch.distributed as dist

from lmdeploy import ChatTemplateConfig, PytorchEngineConfig, pipeline
from lmdeploy.messages import GenerationConfig
from lmdeploy.serve.openai.api_server import handle_torchrun


def set_cuda_visible_device(rank: int):
    devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = devices.split(',')[rank]
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)


if __name__ == '__main__':
    handle_torchrun()
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    set_cuda_visible_device(rank)

    os.environ.setdefault('LMDEPLOY_DP_MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('LMDEPLOY_DP_MASTER_PORT', '29555')

    dist.init_process_group()
    try:
        backend_config = PytorchEngineConfig(
            dp=world_size,
            ep=world_size,
            dp_rank=rank,
        )
        gen_config = GenerationConfig(temperature=1.0, top_k=1, do_sample=True)

        with pipeline(
            '/path/to/model',
            backend_config=backend_config,
            chat_template_config=ChatTemplateConfig(model_name='dsv3-lite'),
            log_level='WARNING',
        ) as pipe:
            outputs = pipe(['hello'], gen_config=gen_config)
            print(outputs[0].text)

        dist.barrier()
    finally:
        dist.destroy_process_group()
```

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=2 dp_ep_pipeline.py
```

Adjust `ep=world_size` for the desired model topology. This exact setting is a useful DP+EP MoE smoke-test shape.
