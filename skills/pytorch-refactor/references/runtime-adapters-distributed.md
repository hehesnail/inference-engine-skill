# Runtime, Adapters, Distributed Context

Use this when touching the public runtime wrapper, mp engine, LoRA adapters,
weight loading, distributed groups, device/backend selection, environment
checks, MoE/EP routing, DeepEP/EPLB, GatedDelta, or NSA indexing.

## Runtime API and MP Engine

Source anchors:

- `engine/engine.py::Engine.create_instance()`
- `engine/engine_instance.py::EngineInstance`
- `engine/mp_engine/__init__.py::build_mp_engine`
- `engine/mp_engine/base.py::MPEngine`, `MPEngineInstance`
- `engine/mp_engine/base_worker.py::EngineInstancePool`,
  `EngineWorkerBase`, `EngineOutputGather`
- `engine/mp_engine/ray_engine.py`, `zmq_engine.py`, `zmq_rpc.py`

Watch points:

- `EngineInstance.async_stream_infer()` owns output offsets, final/cancel
  response assembly, `cache_block_ids`, request metrics, logprobs, and
  optional routed expert transfer object refs for Ray.
- MP engine calls are collective RPCs over workers. Cancellation can arrive
  before a session fully exists; `SessionState` tracks `init_done` and
  `cancelled`. Pre-init cancel should set `cancelled` and make
  `async_stream_infer()` yield `ResponseType.CANCEL`.
- `MPEngineInstance.async_end()` waits for `init_done` and must pop
  `session_states` plus pending-cancel state in a `finally` block, even if the
  backend end RPC fails.
- Ray and ZMQ MP streaming create backend stream tasks. If the client drops the
  frontend generator, the runtime must drop/cancel the abandoned backend task
  and discard any `EngineOutputGather` state for that stream id.
- API endpoints wrap engine result generators through
  `serve/utils/request_cleanup.py`. On normal exit, error, or client
  disconnect, close result generators and remove the request sessions.
- `EngineInstancePool` lazily creates `max_batch_size * 2` instances so a
  sequence can be stopped promptly.
- Keep `sleep()`, `wakeup()`, `update_params()`, schedule metrics, health, and
  P2P methods mirrored across in-process and MP wrappers.
- Engine health is monitored by `serve/core/health.py`. Overlapping backend
  probes return `pending` and should not overwrite the previous snapshot.
  `LMDEPLOY_HEALTH_POLL_INTERVAL`, `LMDEPLOY_HEALTH_PROBE_TIMEOUT`, and
  `LMDEPLOY_HEALTH_UNHEALTHY_AFTER` override monitor timings. Scheduler
  progress uses `scheduler_tick`, which advances once per forward dispatch in
  `InputsMakerAsync._send_next_inputs_impl()`.

Validation:

- `tests/pytorch/engine/test_zmq_rpc.py`
- `tests/pytorch/engine/test_ray_mp_engine.py`
- `tests/pytorch/engine/test_engine_sleep.py`
- `tests/pytorch/engine/test_executor_base.py`
- `tests/test_lmdeploy/serve/test_session_cleanup.py`

## Adapters and Weight Loading

Source anchors:

- `engine/engine.py` builds `AdapterManager`
- `adapter/adapter.py::AdapterManager`, `load_lora_weights()`
- `models/patch.py::add_adapters()`, `build_patched_model()`
- `weight_loader/model_weight_loader.py::ModelWeightLoader`
- `nn/linear/lora.py`, `backends/*/lora.py`, `kernels/cuda/fused_lora.py`

Watch points:

- Adapter ids are deterministic: adapter names are sorted and `None` is id 0.
- Prefix cache roots are partitioned by `adapter_name`; different LoRA adapters
  must not share KV blocks.
- LoRA loading handles packed projections through `packed_modules_mapping` and
  maps weights into per-target `lora_adapters`.
- `ModelWeightLoader` detects safetensors/PyTorch and sharded/unsharded files,
  skips modules marked `_is_dummy_mod`, applies `model.rename_weight()`, calls
  `model.load_weights()`, then calls module `update_weights()` hooks.
- Live `update_params()` splits `mtp.` weights to the draft model when
  speculative method is `qwen3_5_mtp`; preserve this split when changing
  flattened-bucket or IPC update paths.
- Disaggregated weight update exposes PyTorch-only server endpoints
  `/init_weights_update_group`, `/update_weights_from_distributed`, and
  `/destroy_weights_update_group`. The engine serializes these operations,
  creates an external trainer/engine process group with
  `init_custom_process_group()`, supports flattened buckets, splits `mtp.`
  weights to the draft model for `qwen3_5_mtp`, and resets CUDA graphs after a
  finished update because parameter objects can be replaced.
- Qwen3 Omni maps HF `thinker.*` weights into the PyTorch thinker-only model,
  skips `talker.` and `code2wav.` weights, and handles MoE experts plus packed
  QKV/gate-up projections in its local `load_weights()`.
- Custom parameter loading is through `param.weight_loader(param,
  loaded_weight, **kwargs)`. Keep this path intact for sharded, packed,
  quantized, and LoRA weights.

Validation:

- LoRA kernel: `tests/pytorch/kernel/test_fused_lora.py`
- Weight-loader pattern smoke: model-specific focused load test when adding a
  new model or quantization family

## Distributed, Device, Backend, Env

Source anchors:

- `config.py::DistConfig`, `TPMode`
- `engine/config_builder.py::ConfigBuilder`
- `distributed.py::DistContext`, `DistManager`
- `devices/device_manager.py::DeviceManager`
- `backends/selector.py::get_backend()`, `init_backend()`
- `backends/base.py::OpType`, `OpsBackend`
- `envs.py`
- `check_env/*`

Watch points:

- `DistContext.build()` creates CPU world group, attention TP, MLP TP, MoE TP,
  and optional EP groups. Prefer `get_tp_world_rank(layer_type)` and
  `get_dist_group(layer_type)` over assuming one TP group.
- `TPMode.DP_TP` paths may use `StepContext.dp_meta` sizes, especially MoE.
  DP forward metadata is gathered in `engine/model_agent/dp_utils.py`; update
  its schema and tests together when adding scalar DP coordination fields.
- `ModelConfig` stores the resolved `DistConfig`; use
  `ModelConfig.get_num_qkv_head_by_tp()` when backend or graph metadata needs
  TP-local query/KV head counts, including GQA cases where KV heads are
  replicated across TP ranks.
- Speculative decoding now has a separate draft `DistConfig`. `qwen3_5_mtp`
  clones the main dist config; other methods default to draft TP=1.
- Backend selection reads the current `DeviceContext`; direct imports of CUDA
  implementations can break Ascend/MACA/CAMB paths.
- `envs.py` captures env vars read during import through `set_envs()`. Add new
  envs there only when the setting is genuinely global.
- Model-specific `check_env_func` is part of `ModelConfig`; device/model checks
  can live in `configurations/*` and `check_env/*`.

## Operator Anchors: MoE, EP, GatedDelta, NSA

MoE/EP/DeepEP/EPLB:

- `nn/moe/base.py::FusedMoEBase`, `MoEForwardDPTP`, `DispatchInputs`
- `nn/moe/{default,w8a8,blocked_fp8,route}.py`
- `nn/eplb.py::EPLBManager`
- `backends/cuda/moe/*`
- `backends/cuda/token_dispatcher.py::DeepEPTokenDispatcher`
- `backends/deepep_state.py`

GatedDelta/SSM and NSA:

- `nn/gated_delta.py::GatedDeltaMeta`, `GatedDelta`, `CausalConv1d`,
  `load_state()`
- `backends/cuda/gated_delta_rule.py`
- `backends/gated_delta_rule.py::GatedDeltaRuleImpl.prepare_inputs()`
- `kernels/cuda/gated_delta_preprocess.py`
- `kernels/cuda/gated_delta_rule.py`
- `nn/nsa.py::IndexerTopKFP8`
- `backends/cuda/nsa.py::TritonNSAIndexFP8`

Watch points:

- MoE layer type may be `attn`, `mlp`, or `moe` for TP queries. Do not use the
  default TP group for MoE unless the code already does so.
- DeepEP low-latency mode depends on decoding state; normal/prefill and decode
  paths can use different dispatch implementations.
- `DeepEPState.enabled()` is gated by EP mode; enabling the state flag alone
  should not activate DeepEP when `dist_config.ep == 1`.
- EPLB changes logical/physical expert ids; check dispatch info and expert-list
  assumptions before modifying routing.
- `GatedDeltaMeta` reads `StepContext` and attention metadata to build state
  ids, conv offsets, cache seqlens, and spec offsets. Spec decode and prefill
  offsets are easy to desynchronize.
- Qwen3.5/Qwen3-Next recurrent state layout is transposed to
  `(num_delta_layers, num_v_heads, head_v_dim, head_k_dim)`; MTP adds the spec
  slot dimension before heads. GatedDelta kernels receive
  `transpose_state_layout=True` for this layout.
- Recurrent GatedDelta kernels support `state_indices`, circular
  `cache_seqlens`, and invalid state ids. Invalid ids must zero the output and
  avoid mutating the state bank.
- GatedDelta q/k/g/beta preparation is now owned by
  `GatedDeltaRuleImpl.prepare_inputs()`. CUDA may fuse q/k repeat,
  q/k l2norm, beta sigmoid, g softplus, and init-token masking via
  `gated_delta_preprocess`; the caller must pass
  `use_qk_l2norm_in_kernel and not qk_l2norm_done` into the chunk/recurrent
  kernels.
- The fused GatedDelta preprocess kernel supports non-power-of-2 `kv_ratio`
  values by using a power-of-2 `BLOCK_KV_RATIO` plus masks for dt/a-scale and
  beta/g stores. Keep `kv_ratio=3` coverage when editing this path.
- TileLang kernel code has to tolerate version differences such as
  `T.shfl_sync` argument order and may need explicit pass config such as
  `TL_DISABLE_DATA_RACE_CHECK`.
- NSA FP8 indexing intentionally uses allocated cache length for decode
  cudagraph compatibility.

Validation:

- MoE kernels: `tests/pytorch/kernel/test_fused_moe.py`,
  `tests/pytorch/kernel/test_fuse_moe_blocked_fp8.py`,
  `tests/pytorch/kernel/test_moe_route.py`
- GatedDelta/NSA: add model- or kernel-focused tests when editing; broad
  `tests/pytorch/kernel/` may not cover every state-cache path. Include
  `tests/pytorch/kernel/test_gated_delta_preprocess.py` when touching fused
  preprocess.
