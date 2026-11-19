---
name: lmdeploy-pipeline
description: Use LMDeploy's Python pipeline and self-hosted serving for generation, chat sessions, adapter selection, and scoring. Start a server or proxy, verify client-visible generation, and diagnose multi-worker or split prefill/decode deployments.
---

# LMDeploy Pipeline and Serving

For usage questions, give the command or client request for the user's LMDeploy version. When asked to start or verify a service, send a small request through the intended client endpoint and check its result. For a new checkpoint with a supported architecture, start with a normal pipeline generation. Use `pytorch-refactor` when the architecture fails to resolve or the implementation needs to change. When the task is to compare throughput, explain latency, or interpret a trace, carry this LMDeploy service's effective settings, request IDs, and worker records into the shared measurement guides in `inference-engine`.

## Choose the Reference

- To generate, stream, continue or close a [chat session](references/pipeline-api.md#chat-sessions), pass images, or score inputs in Python, use the [pipeline API](references/pipeline-api.md) for `GenerationConfig` and backend options. For perplexity, check how the checkpoint and token IDs were chosen. For HTTP scoring, use the [perplexity endpoint](references/serving.md#perplexity-endpoint).
- To start one server or reproduce a client request, use [serving](references/serving.md). Select the intended model ID and client route. For per-request LoRA selection, use the [registered chat adapter alias](references/serving.md#lora-adapter-selection). A real generation through that endpoint checks what the client can use.
- When several workers serve through one proxy, one worker gets no traffic, placement is unexpected, or traces show no forward work, use [DP/EP serving](references/dp-ep.md) for `--dp`/`--ep`, DeepEP, and `torchrun` options. For an existing trace, match the request time to the worker and its recorded capture window.
- When prefill and decode run on separate workers, a registered model can still stall before generation. Use the [PD/DistServe smoke test](references/pd-disagg.md#smoke-test) to check connection warmup and a multi-token response through the proxy. If warmup succeeds but handoff stalls, trace KV migration in `pytorch-refactor`.

## Quick Pipeline

```python
from lmdeploy import pipeline

pipe = pipeline('/path/to/model')
response = pipe('Hello, who are you?')
print(response.text)
```

## Quick Serve

Run in the server environment allocated to the job. On a shared server without an allocation, check availability and set `CUDA_VISIBLE_DEVICES` to the selected GPU before launching.

```bash
lmdeploy serve api_server /path/to/model --server-port 8100
```

An OpenAI SDK client targeting this LMDeploy server uses `http://HOST:PORT/v1`. Use a model ID returned by that server’s `/v1/models`. For Anthropic-compatible `/v1/messages`, use `http://HOST:PORT` because the SDK appends `/v1`.

## Quick DP/EP Serve Shape

[DP/EP serving](references/dp-ep.md) covers worker placement, profiler traces, and port selection.

1. Export `LMDEPLOY_DP_MASTER_ADDR` and `LMDEPLOY_DP_MASTER_PORT`.
2. Start `lmdeploy serve proxy` as one long-lived process.
3. Start `lmdeploy serve api_server MODEL --backend pytorch --dp DP --tp TP --ep EP --proxy-url PROXY_URL` as another long-lived process.
4. Treat `ray start` as optional; use it only when an explicit Ray head, a fresh cluster, or custom Ray resources are needed.

Clients normally call the proxy port when using the DP/EP serve path.

## Quick PD/DistServe Shape

[PD/DistServe](references/pd-disagg.md) has the per-role launch and connection warmup commands.

1. Start `lmdeploy serve proxy --serving-strategy DistServe`.
2. Start one or more prefill API servers with `--role Prefill --backend pytorch --proxy-url PROXY_URL`.
3. Start one or more decode API servers with `--role Decode --backend pytorch --proxy-url PROXY_URL`.
4. Warm or verify `/distserve/connection_warmup`, then send client requests to the proxy.

Keep PyTorch PD roles on `block_size == kernel_block_size`; PD migration does not support split logical/kernel cache blocks.

## Bundled Samples

- `samples/rest_api_chat.py`: OpenAI-compatible `v1/chat/completions` client.
- `samples/rest_api_messages.py`: Anthropic-compatible `v1/messages` client.

## Source Files

| File | Purpose |
|------|---------|
| `lmdeploy/api.py` | Top-level `pipeline()` factory function |
| `lmdeploy/pipeline.py` | `Pipeline` implementation |
| `lmdeploy/messages.py` | `GenerationConfig`, `TurbomindEngineConfig`, `PytorchEngineConfig`, `Response` |
| `lmdeploy/model.py` | `ChatTemplateConfig` |
| `lmdeploy/serve/processors/multimodal.py` | `MultimodalProcessor` for VLM input formatting |
