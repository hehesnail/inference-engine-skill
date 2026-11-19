# Disaggregated Serving / PD Migration

Use this for PyTorch DistServe, prefill/decode roles, P2P KV migration, or
proxy/engine connection logic. Read `cache-engine.md` too when block layout or
cache size is involved.

## Scope

- Public config: `PytorchEngineConfig.role`, `migration_backend`, and
  `enable_transfer_obj_ref`.
- Core enums/configs: `lmdeploy/pytorch/disagg/config.py`
  (`ServingStrategy`, `EngineRole`, `MigrationBackend`, RDMA/TCP/NVLink config).
- Backends: `disagg/backend/dlslime.py` and `disagg/backend/mooncake.py`.
- Connections: `disagg/conn/engine_conn.py` for engine-side P2P RPC and
  `disagg/conn/proxy_conn.py::PDConnectionPool` for proxy-side prefill/decode
  pairing.

## Engine Flow

1. `Engine.__init__` creates `EngineP2PConnection(self)` and stores migration
   events when async loops start.
2. Non-Hybrid roles require `EngineLoop(..., engine_conn=...)`; the loop starts
   `migration_loop()` in addition to the main loop.
3. Prefill completion can return `cache_block_ids`; decode-side requests carry
   `migration_request` data and enter scheduler migration states.
4. `EngineLoop.migration_loop()` calls `Scheduler._schedule_migration()`,
   then `_migration_loop_migrate()`.
5. `_migration_loop_migrate()` maps remote prefill block ids to local decode
   block ids and calls `executor.migrate(MigrationExecutionBatch)`.
6. After migration, `engine_conn.zmq_send()` notifies the remote engine and
   `_migration_loop_get_outputs()` emits the migrated token response.

## Cache Engine Contract

- `CacheEngine.p2p_initialize()` creates the migration backend, registers
  `full_gpu_cache` as memory region metadata, and returns per-rank transfer
  endpoint info.
- `CacheEngine.p2p_connect()` connects to the matching remote rank endpoint.
- `CacheEngine.migrate()` builds one `AssignmentInstruct` per layer and block
  pair using the local and remote `num_gpu_blocks` stride.
- PD migration currently requires `cache_config.block_size ==
  cache_config.kernel_block_size`; `engine_checker.py` should reject unsupported
  layouts before runtime migration.

## SSM / State-Cache Status

Current PD migration is KV-block oriented. SSM/state-cache models such as
Qwen3.5 are not covered by the existing PD contract yet. Do not add or assume
SSM PD support without first designing:

- what decode receives after prefill: migrated state cache, recomputed state, or
  an exact ready-state checkpoint;
- ownership of `state_caches`, logical state ids, `state_offsets`, and any
  prefix-cache state checkpoints across prefill and decode engines;
- scheduler migration admission/allocation for both KV blocks and state slots,
  including rollback and cache-free behavior;
- validation that covers P2P init/connect, KV migration, state correctness, and
  multi-request DP routing for an SSM model.

## Watch Points

- `Scheduler._schedule_migration()` has its own admission and eviction path.
  Verify prefix-cache matching and block allocation there separately from normal
  prefill scheduling.
- Migration uses physical byte offsets in the packed cache pool. If cache
  packing, quant cache descs, or logical/kernel block mapping changes, update
  migration offset/stride calculations together.
- Sleep drains both main and migration loops for non-Hybrid roles. Do not let
  stale prefetched work survive a sleep/wakeup boundary.

## Validation

- Unit: `tests/pytorch/engine/test_cache_engine.py` and
  `tests/pytorch/engine/test_engine_checker.py` for block-layout rejection.
- Focused smoke: one Prefill role engine and one Decode role engine through the
  proxy path, verifying P2P init/connect, migration, and cache-free behavior.
