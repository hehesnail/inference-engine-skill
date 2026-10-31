<!--
SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Source-Backed Tuning Notes

Read an exact or nearby checked-in config and the model's deployment guide **before** using these notes. These are not universal thresholds.

## Commonly Tuned Fields

| Field | Guidance |
|---|---|
| `max_batch_size` | Scheduler ceiling, not a memory reservation and NOT proportional to concurrency — actual batch size adapts at runtime. Copy from the nearest checked-in source config; do not invent a value from concurrency. Prefer keeping the source value unless OOM occurs. MoE models generally cap lower than dense. |
| `max_num_tokens` | Scheduler token budget. When chunked prefill is **disabled** (default): must cover ISL plus chat template overhead; an ISL-to-2×ISL range is only a workload-specific starting hypothesis. When chunked prefill is **enabled**: bounds the aggregate scheduled tokens per iteration (each request's chunk depends on the scheduler and concurrent decode work) — see `enable_chunked_prefill` section below. General default is 8192. Tune together with `max_batch_size`. |
| `max_seq_len` | Global hard cap on total tokens per request (prompt + output). Set to `ISL + OSL + chat_template_overhead`. Chat templates and benchmarking preambles add tokens beyond raw ISL — overhead varies by model (checked-in configs show 20–200 tokens). Setting too tight rejects or truncates requests. Paged KV is not necessarily preallocated to this cap per request; inspect actual block allocation, graph/workspace sizes and admission policy before attributing memory to a loose cap. Copy from nearest checked-in config when available. |
| `enable_attention_dp` | High-throughput knob. MoE+GQA models benefit at lower concurrency thresholds than MoE+MLA or Dense+GQA. Rank-local attention/KV layout changes; MLA and GQA can have different memory costs. Attention DP can distribute requests, so do not infer that each request is replicated on every rank. Can trigger OOM when combined with aggressive KV cache fraction. Follow the exact model guide/config. |
| `kv_cache_config.free_gpu_memory_fraction` | OOM lever. MLA models (compressed KV) tolerate higher fractions; GQA models need more headroom. Lower when ADP enabled to account for replicated attention overhead. Large MoE models with ADP may need notably conservative fractions. Guides often adjust `max_batch_size` or `max_seq_len` first. |
| `moe_expert_parallel_size` / `moe_config.backend` | MoE only. Copy both from checked-in source — EP does not necessarily equal TP. If no backend source exists, mark as unverified; benchmark CUTLASS vs TRTLLM. |
| `cuda_graph_config.max_batch_size` / `batch_sizes` | Caps which decode batch sizes get CUDA graphs captured; batches above this fall back to eager execution (no error, just slower). Use the source recipe and expected reachable decode sizes; capturing every scheduler size can consume substantial memory and is not universally safe. Consider lowering when memory is tight — e.g., DeepSeek-R1 conc=1 uses `cuda_graph_config.max_batch_size: 1` with server `max_batch_size: 512` to avoid wasting graph memory on unreachable sizes. Also capped by `max_num_tokens / (1 + max_total_draft_tokens)` at runtime. |

## KV Cache Estimation

Use these formulas to sanity-check whether a concurrency target fits in GPU memory. Read the required values from the model's HuggingFace config (`config.json`).

**Per-token KV cache size:**

- **GQA (standard grouped-query attention):**
  `kv_per_token_per_rank = 2 × local_attention_layers × local_kv_heads × head_dim × kv_dtype_bytes`
  At f405e10b730e, `Attention.__init__` uses `local_kv_heads = (num_key_value_heads + attention_tp_size - 1) // attention_tp_size` for its supported layout. In standard divisible layouts this is the quotient, with at least one local KV head when TP exceeds the KV-head count; heads do not become fractional. A capacity formula does not establish support for arbitrary non-divisible shapes: check weight partitioning and backend constraints. Under attention DP, KV is not attention-TP-sharded inside a replica, but requests may be partitioned across replicas: use the actual request placement rather than assuming every request is duplicated everywhere.
- **MLA (multi-latent attention, e.g. DeepSeek-V2/V3):**
  `kv_per_token_per_rank = local_attention_layers × (kv_lora_rank + qk_rope_head_dim) × kv_dtype_bytes` for the standard latent-plus-RoPE layout; compressed/quantized variants require their actual pool layout and metadata

Where `kv_dtype_bytes` is 2 for BF16/FP16, 1 for FP8/INT8, before format-specific scale/metadata overhead.

**Approximate max concurrent requests (upper bound):**

```
max_requests_memory_bound ≈ floor(measured_available_KV_bytes_per_rank /
                                  (kv_per_token_per_rank × (ISL + OSL)))
```

Use rank-local weight placement and measured post-load/runtime/graph reservations. A fixed 10% total-HBM reserve or total weights divided by TP cannot represent PP/EP/ADP, unsharded tensors or offload. This is a capacity upper bound, not latency/SLO concurrency; account for page rounding, prefix sharing, state pools and skew.

**HF config fields to read:** `num_attention_layers` (equals `num_hidden_layers` for standard transformers; differs for hybrid models like Nemotron-H), `num_key_value_heads`, `head_dim` (or `hidden_size / num_attention_heads`), `kv_lora_rank`, `qk_rope_head_dim`.

**Caveats:** This estimate ignores activation memory, CUDA graph workspace, MoE expert workspace, and attention data parallelism (ADP) overhead. Always prefer checked-in config values over formula-derived estimates. Mark any formula-derived number as unverified.

## Chunked Prefill

Chunked prefill (`enable_chunked_prefill: true`) splits long prefill sequences into chunks so that decode batches sharing the same iteration are not starved. It is **disabled by default** and should be treated as an advanced latency optimization, not a default recommendation. See the `max_num_tokens` table entry above for how it changes token budget semantics.

**MLA models (DeepSeek-V2/V3/R1, Kimi-K2):**
- Chunked prefill IS supported for MLA — dedicated CUDA kernels exist with multi-round attention and softmax merging.
- **Version/backend constraint:** inspect the actual MLA dispatcher. At f405e10b730e, `MLA.forward_context` selects its dedicated chunked-prefill path only on SM100+ and uses a cached-KV fallback for other supported paths; FP4 cached context has an additional SM100+/chunking requirement. This does not establish support for every Blackwell SKU or every attention backend.
- **Trade-off:** chunking changes prefill/decode interleaving; measure TTFT, TPOT and throughput for the target load. A historical throughput trade-off is not a guarantee that chunking always lowers throughput.
- **Recommendation:** do not enable by default for MLA models. Consider it only for latency-sensitive workloads on Hopper or Blackwell GPUs where TPOT reduction outweighs the throughput cost.

**Non-MLA models (GQA):** more broadly supported across GPU generations. Still disabled by default; enable when long prefill sequences cause decode latency spikes.
