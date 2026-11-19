# PD / DistServe Disaggregated Serving Reference

Use this reference when the user asks how to deploy or smoke-test LMDeploy PD
disaggregated serving, DistServe, Prefill/Decode roles, P2P KV migration,
DLSlime/RDMA transfer, or proxy connection warmup. For engine internals or code
changes, use `pytorch-refactor` and read its `references/disagg-pd.md`.

## Serve Startup Shape

PD serving uses one proxy plus separate PyTorch API servers for prefill and
decode. Clients call the proxy port. The four-GPU example below uses two disjoint pairs. Substitute the devices assigned to each role in your job.

```bash
export MODEL_PATH=/path/to/model
export MODEL_NAME=served-model

export PROXY_HOST=127.0.0.1
export PROXY_PORT=18100
export PROXY_URL=http://${PROXY_HOST}:${PROXY_PORT}

export DP=2
export EP=2
export TP=1
export MAX_BATCH_SIZE=32
export MAX_PREFILL_TOKEN_NUM=4096
export CACHE_MAX_ENTRY_COUNT=0.55
export CACHE_BLOCK_SEQ_LEN=64

# Useful for MoE DP/EP capacity planning.
export DEEPEP_MAX_BATCH_SIZE=512
export DEEPEP_MAX_TOKENS_PER_RANK=512
```

This smoke test expects fresh proxy state. If the shared installation contains registrations from another job, use a run-owned installation. The proxy can load its existing `proxy_config.json` at import time, before `--disable-cache-status` takes effect.

```bash
lmdeploy serve proxy \
  --server-name ${PROXY_HOST} \
  --server-port ${PROXY_PORT} \
  --routing-strategy min_expected_latency \
  --serving-strategy DistServe \
  --migration-protocol RDMA \
  --link-type RoCE \
  --disable-cache-status \
  --log-level INFO
```

Prefill API server process:

```bash
export LMDEPLOY_DP_MASTER_ADDR=127.0.0.1
export LMDEPLOY_DP_MASTER_PORT=29555
export CUDA_VISIBLE_DEVICES=0,1

lmdeploy serve api_server ${MODEL_PATH} \
  --backend pytorch \
  --model-name ${MODEL_NAME} \
  --dp ${DP} \
  --tp ${TP} \
  --ep ${EP} \
  --max-batch-size ${MAX_BATCH_SIZE} \
  --max-prefill-token-num ${MAX_PREFILL_TOKEN_NUM} \
  --cache-max-entry-count ${CACHE_MAX_ENTRY_COUNT} \
  --cache-block-seq-len ${CACHE_BLOCK_SEQ_LEN} \
  --proxy-url ${PROXY_URL} \
  --role Prefill \
  --nnodes 1 \
  --node-rank 0
```

Decode API server process:

```bash
export LMDEPLOY_DP_MASTER_ADDR=127.0.0.1
export LMDEPLOY_DP_MASTER_PORT=29655
export CUDA_VISIBLE_DEVICES=2,3

lmdeploy serve api_server ${MODEL_PATH} \
  --backend pytorch \
  --model-name ${MODEL_NAME} \
  --dp ${DP} \
  --tp ${TP} \
  --ep ${EP} \
  --max-batch-size ${MAX_BATCH_SIZE} \
  --max-prefill-token-num ${MAX_PREFILL_TOKEN_NUM} \
  --cache-max-entry-count ${CACHE_MAX_ENTRY_COUNT} \
  --cache-block-seq-len ${CACHE_BLOCK_SEQ_LEN} \
  --proxy-url ${PROXY_URL} \
  --role Decode \
  --nnodes 1 \
  --node-rank 0
```

## Explicit Per-Rank Launcher

If the CLI launcher cannot express the desired GPU/rank mapping, launch one
process per DP rank and set `PytorchEngineConfig.dp_rank` directly. This is
useful for bounded smoke tests and for avoiding accidental use of extra GPUs.

```python
import os

from lmdeploy.messages import PytorchEngineConfig
from lmdeploy.pytorch.disagg.config import EngineRole, MigrationBackend
from lmdeploy.serve.openai.api_server import serve

role = os.environ['ENGINE_ROLE']  # Prefill or Decode
dp_rank = int(os.environ['DP_RANK'])

os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['GPU_ID']
os.environ['LMDEPLOY_DP_MASTER_ADDR'] = os.environ['DP_MASTER_ADDR']
os.environ['LMDEPLOY_DP_MASTER_PORT'] = os.environ['DP_MASTER_PORT']

backend_config = PytorchEngineConfig(
    dp=int(os.environ.get('DP', '2')),
    ep=int(os.environ.get('EP', '2')),
    tp=int(os.environ.get('TP', '1')),
    dp_rank=dp_rank,
    max_batch_size=int(os.environ.get('MAX_BATCH_SIZE', '32')),
    max_prefill_token_num=int(os.environ.get('MAX_PREFILL_TOKEN_NUM', '4096')),
    cache_max_entry_count=float(os.environ.get('CACHE_MAX_ENTRY_COUNT', '0.55')),
    block_size=int(os.environ.get('CACHE_BLOCK_SEQ_LEN', '64')),
    role=EngineRole[role],
    migration_backend=MigrationBackend.DLSlime,
    distributed_executor_backend='ray',
    enable_metrics=False,
)

serve(
    os.environ['MODEL_PATH'],
    model_name=os.environ.get('MODEL_NAME'),
    backend='pytorch',
    backend_config=backend_config,
    server_name=os.environ.get('SERVER_HOST', '127.0.0.1'),
    server_port=int(os.environ['SERVER_PORT']),
    proxy_url=os.environ['PROXY_URL'],
    log_level='INFO',
)
```

## Optional Ray Pre-Start

Ray is required for `dp > 1`. Let LMDeploy initialize Ray unless a test needs an
explicit fresh cluster or a hard GPU cap. Start the following head in an isolated job environment. Stop an existing head only if it belongs to this run.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ray start \
  --head \
  --num-cpus 4 \
  --num-gpus 4 \
  --port 0 \
  --object-store-memory 34359738368
```

## Smoke Test

After all prefill and decode nodes register, warm up P2P connections:

```bash
curl -H 'Content-Type: application/json' \
  -X POST ${PROXY_URL}/distserve/connection_warmup \
  -d '{}'
```

Expected response:

```json
{"SUCCESS": true}
```

Then send a request to the proxy:

```bash
curl ${PROXY_URL}/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "'"${MODEL_NAME}"'",
    "prompt": "Write one short sentence about Shanghai.",
    "temperature": 0,
    "max_tokens": 16,
    "stream": false
  }'
```

A completion with the expected model and generated output verifies the client path through the proxy. To verify coverage of every DP rank, send concurrent requests and inspect worker logs, per-rank counters, or traces that identify which ranks handled the requests. Record the observed ranks for each prefill and decode group. The router can select the same worker more than once.

## Notes And Gotchas

- Only the PyTorch backend supports PD/DistServe.
- PD roles are `Hybrid`, `Prefill`, and `Decode`; use `Prefill` and `Decode`
  for disaggregated serving.
- PD migration currently requires `block_size == kernel_block_size`. If passing
  `--cache-block-seq-len`, do not set a different `--kernel-block-size`.
- Prefix caching is disabled for PD/disaggregated roles unless that path is
  explicitly redesigned and tested.
- Default migration backend is `DLSlime`; the practical tested protocol is
  RDMA with GDR. NVLink may fall back to RDMA depending on the installed
  DLSlime build.
- Use distinct `LMDEPLOY_DP_MASTER_PORT` values for prefill and decode groups
  when they are separate DP process groups.
- A 2026-06-24 smoke in `lmdeploy_deepep_yq` ran Qwen3 30B A3B Thinking on four H200s with prefill GPUs `0,1`, decode GPUs `2,3`, `dp=2`, `ep=2`, and `tp=1`.
