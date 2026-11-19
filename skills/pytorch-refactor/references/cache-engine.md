# LMDeploy PyTorch Cache Engine Notes

Use this reference when changing or reviewing `lmdeploy/pytorch/engine/cache_engine.py`,
`lmdeploy/pytorch/engine/executor/base.py`, or cache-related input maker logic.

## Cache Sizing

- `ExecutorBase.update_configs()` runs after model build and before cache engine build.
- Free memory is collected per worker/rank; final `num_gpu_blocks` must fit all ranks.
- State cache is a separate pool, not pageable KV cache. Subtract state cache memory from each rank before KV sizing.
- For SSM prefix caching, `prefix_cache_state_budget` adds extra checkpoint
  slots on top of active request state slots and the reserved system row. It is
  not a replacement for runtime state capacity.
- Runtime buffers are not pageable KV cache. Estimate and reserve runtime memory before applying `cache_max_entry_count`.
- Apply `cache_max_entry_count` only to the remaining memory intended for pageable KV cache.
- `num_gpu_blocks != 0` means the user supplied an explicit block count. Do not resize it from a current free-memory snapshot.
- Spec decode draft/spec cache sizing depends on draft attention TP from
  `SpecDecodeConfig.dist_config`:
  - draft TP = 1: draft/spec cache is only built on one rank in each target
    attention-TP group; that spec rank uses target KV block + spec KV block,
    while non-spec ranks use target KV block only.
  - draft TP > 1: draft/spec cache is sharded across the participating draft TP
    ranks, so every participating rank carries the sharded target + spec
    footprint.
  Use `ExecutorBase._get_spec_attn_tp()` and take the minimum block capacity
  across ranks.

## Block Size Invariants

- `block_size` is the logical scheduler block size.
- `kernel_block_size` is the physical kernel cache block size.
- `block_size` must be greater than or equal to `kernel_block_size` and divisible by it.
- `CacheEngine.allocate_caches()` stores `num_gpu_blocks * (block_size // kernel_block_size)` physical kernel blocks.
- `engine/inputs_maker.py` maps logical block ids to kernel block ids. Keep this mapping aligned with cache allocation.
- When executor logic adjusts target cache block sizes, keep spec cache `block_size` and `kernel_block_size` aligned with the final target cache layout.

## Prefix Cache Boundaries

- Prefix cache uses scheduler logical blocks, not raw cache tensors. `BlockTrie`
  stores logical block ids and relies on `BaseBlockManager.get_block_table()` to
  map them to physical kernel blocks.
- `BlockTrie.match()` must run before cache allocation so matched prefix blocks
  reduce the suffix that needs new blocks. `BlockTrie.allocate()` must run after
  allocation so newly completed full blocks are registered and ref-counted.
- Prefix caching supports AR text plus narrow PyTorch VLM and SSM paths, and
  AR-spec/MTP through a private recompute-overlap block. Sliding-window
  attention, PD/disaggregated roles, TurboMind VLM, and legacy PyTorch VLM
  remain disabled unless redesigned.
- VLM block keys include multimodal content identity from
  `PrefixCacheMeta`/`MultiModalData.content_hash`; identical placeholder tokens
  with different media must not share KV.
- SSM prefix caching requires exact ready state checkpoints. A KV-only SSM hit
  is invalid.
- `prefix_cache_decode_state_interval=0` disables decode checkpoint saves; a
  positive interval must be a multiple of `block_size`.
- Split logical/kernel blocks require `InputsMakerAsync._map_to_kernel_block_offsets()`
  to stay consistent with `CacheEngine.allocate_caches()`.

## PD Migration

- PD migration currently assumes one logical block maps to one kernel block.
- Therefore PD migration requires `block_size == kernel_block_size`.
- If supporting split kernel blocks in PD migration later, update offset, length, and layer-stride calculations together and add migration tests.
- Current PD migration covers KV cache movement only. SSM/state-cache PD support
  needs an explicit design for state handoff, state-slot allocation, rollback,
  and correctness validation before enabling models such as Qwen3.5.
- DistServe/PD roles should reject unsupported block layouts in `engine_checker.py`, before runtime migration.
- DisAgg code lives under `lmdeploy/pytorch/disagg/`; cache movement still
  bottoms out in `CacheEngine.p2p_initialize()`, `p2p_connect()`, and
  `migrate()`.
- Migration backends are selected by `MigrationBackend` (`DLSlime` or
  `Mooncake`) and operate on registered `full_gpu_cache` memory regions.

## Cache Allocation

- KV and custom cache descs are packed into one contiguous `uint8` memory pool.
- The pool is zero-initialized intentionally. This avoids abnormal outputs if a kernel accidentally reads an unwritten cache region.
- Custom cache shapes from `model_config.cache_shapes` should go through pooled cache descriptors; avoid reintroducing a separate custom-cache allocation path unless there is a concrete user.
- The cache engine uses the canonical `"cuda"` device path. Non-CUDA integrations currently patch or shim this path in their backend layer, so do not replace it casually with a raw `device_type` string.

## State Cache

- State cache is allocated by `StateCacheEngine` as a separate pool.
- `paging/state_manager.py::build_state_manager()` reserves logical state id 0
  for system/padding use. `CacheConfig.num_state_caches` is the total allocated
  state-cache row count including that reserved row, so allocatable sequence
  ids are `1..num_state_caches-1`.
- Non-SSM paths with `num_state_caches is None` have zero allocatable states;
  they should not accidentally allocate state id 0.
- State-space models can use prefix caching only through exact trie-owned state
  checkpoints. Runtime state slots are stored on `seq.logical_state`; frozen
  checkpoint slots live on trie nodes.
- Current main has no `StateCacheEngine.init_caches()` hook in `model_forward()`.
  State cache tensors are passed into `StepContext`; model code receives
  `state_offsets`/`state_ids` and updates state in its own forward path.
- `StateCacheEngine.copy_caches()` copies non-overlapping state slots from
  compact host integer src/dst pairs and coalesces contiguous runs. It is used
  on the model forward stream to restore `checkpoint -> runtime_state` before
  forward and save `runtime_state -> checkpoint` after forward.
- CUDAGraph SSM handling pads `state_ids` through `CudaGraphMixin` buffers and
  rewrites `context.state_offsets` during capture/replay.

## Sleep / Wakeup

- Normal wakeup rebuilds cache engines from the existing config.
- Waking all tags, or waking `kv_cache`, rebuilds cache engines and runs
  warmup inside `BaseModelAgent.wakeup()` before `state.is_sleeping=False` and
  before DP `to_wakeup` is released. The top-level `Engine.wakeup()` delegates
  to the executor, clears sleeping tags, and unblocks new inputs only after all
  tags clear.
- In MP executor, `wakeup(['kv_cache'])` and `wakeup(None)` wait for worker
  return values; `wakeup(['weights'])` can be fire-and-forget.
- `empty_init` skips init-time warmup, but wakeup and update-params paths must
  still leave graph runners/cache engines ready before serving work resumes.
- Avoid resetting `num_gpu_blocks` to 0 after sleep and then re-estimating from free memory; weights/cache may not have been restored yet, which can overestimate available cache memory.
