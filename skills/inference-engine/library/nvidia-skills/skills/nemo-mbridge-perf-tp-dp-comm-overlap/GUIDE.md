> 语义适用性核对：这是 Megatron Bridge 训练参考。TP/PP通信重叠配置必须与runtime启用条件、环境接线对照。 optimizer/gradient/backward/activation-recompute 配置不是推理 KV cache、continuous batching 或 serving DP 的实现规则；具体字段以目标 Bridge checkout 为准。

> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[NVIDIA/skills@25340619336e](https://github.com/NVIDIA/skills/blob/25340619336e89bcc40365c28bfe9e790430468e/skills/nemo-mbridge-perf-tp-dp-comm-overlap/SKILL.md)，2026-10-31 核对。

# TP / DP / PP Communication Overlap Skill

For stable background and recommendation level, see:

- @docs/training/communication-overlap.md

## Enablement

Minimal Bridge override:

```python
from megatron.bridge.training.comm_overlap import CommOverlapConfig

cfg.model.tensor_model_parallel_size = 4
cfg.model.sequence_parallel = True
cfg.model.pipeline_model_parallel_size = 4
cfg.model.virtual_pipeline_model_parallel_size = 2

cfg.comm_overlap = CommOverlapConfig(
    tp_comm_overlap=True,
)

cfg.ddp.use_distributed_optimizer = True
cfg.ddp.overlap_grad_reduce = True
cfg.ddp.overlap_param_gather = True
```

Optional TP preset:

```python
from megatron.bridge.training.comm_overlap import userbuffers_bf16_h100_h12288_tp4_mbs1_seqlen2048

cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_h100_h12288_tp4_mbs1_seqlen2048
```

Precision knobs belong to mixed precision:

```python
cfg.mixed_precision.grad_reduce_in_fp32 = False
cfg.mixed_precision.fp8_param_gather = False
```

## Code Anchors

Bridge overlap gating:

```439:449:src/megatron/bridge/training/comm_overlap.py
if self.user_comm_overlap_cfg.tp_comm_overlap is True:
    if model_cfg.tensor_model_parallel_size < 2:
        ...
    elif not model_cfg.sequence_parallel:
        ...
    elif not HAVE_TE:
        ...
```

PP overlap selection:

```451:458:src/megatron/bridge/training/comm_overlap.py
if model_cfg.pipeline_model_parallel_size > 1:
    if vp_size > 1:
        comm_overlap_cfg.overlap_p2p_comm = True
        comm_overlap_cfg.batch_p2p_comm = False
    else:
        comm_overlap_cfg.overlap_p2p_comm = False
        comm_overlap_cfg.batch_p2p_comm = True
```

DP overlap defaults:

```572:579:src/megatron/bridge/training/comm_overlap.py
if self.data_parallel_size > 1:
    comm_overlap_cfg.bucket_size = 128 * 1024 * 1024
    comm_overlap_cfg.overlap_grad_reduce = True
    comm_overlap_cfg.overlap_param_gather = True
```

Launch-time env tuning:

```570:609:src/megatron/bridge/recipes/run_plugins.py
executor.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] = str(cuda_device_max_connections)
...
executor.env_vars["NVTE_FWD_LAYERNORM_SM_MARGIN"] = str(self.layernorm_sm_margin)
executor.env_vars["NVTE_BWD_LAYERNORM_SM_MARGIN"] = str(self.layernorm_sm_margin)
```

## Pitfalls

1. TP overlap silently disables itself if `sequence_parallel=False` or Transformer Engine is unavailable.
2. PP overlap is not enabled for all PP cases. Bridge only auto-selects `overlap_p2p_comm=True` when `PP > 1` and `VPP > 1`.
3. `bucket_size` is a parameter-count knob, not a byte-size knob.
4. `grad_reduce_in_fp32` and `fp8_param_gather` should be set through mixed precision, not as standalone DDP tuning first.
5. `CUDA_DEVICE_MAX_CONNECTIONS` and LayerNorm SM margin are launch-time plugin settings, not `CommOverlapConfig` fields.

## Verification

Use the checked-in overlap unit coverage first:

```bash
uv run python -m pytest tests/unit_tests/training/test_comm_overlap.py -q
```

Optional second check if `nemo_run` is available:

```bash
uv run python -m pytest tests/unit_tests/recipes/test_run_plugins.py -q
```

Success criteria:

- first command reports `26 passed`
- second command validates plugin-owned env wiring when not skipped
