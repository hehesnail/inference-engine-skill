# Prefix Caching

Use this when changing PyTorch prefix-cache reuse, trie/block/state-manager
behavior, VLM multimodal hashes, SSM state checkpoints, routed-expert replay,
cached-token accounting, or prefix-cache metrics. Do not load long-context
chunking notes unless the change also touches `LongContextChunker`.

## Current Scope

- Prefix caching supports AR text reuse, AR-spec/MTP reuse with an overlap
  recompute block, plus narrow PyTorch VLM and SSM paths.
- VLM support is limited to PyTorch new-preprocess `HistoryMultiModals` paths.
  TurboMind VLM, legacy PyTorch VLM, and `InputEmbeddings`-only VLM paths are
  still out of scope.
- SSM/state-cache prefix caching is opt-in behind `enable_prefix_caching=True`
  and requires exact ready state checkpoints. KV-only SSM prefix hits are
  invalid.
- PD/disaggregated-serving roles and sliding-window attention remain
  disabled/out of scope. Spec decoding/MTP is no longer blanket-disabled, but
  currently relies on the AR-spec recompute-overlap contract below.
- Sliding-window attention disables it in
  `CacheConfig.__post_init__()`/`ExecutorBase.__init__()`.
- Public config path: `PytorchEngineConfig.enable_prefix_caching` ->
  `ConfigBuilder.build_cache_config()` -> `CacheConfig.enable_prefix_caching`.
- SSM config fields:
  - `prefix_cache_state_budget` is extra checkpoint capacity; active runtime
    state slots plus one reserved row are still allocated separately.
  - `prefix_cache_decode_state_interval=0` disables decode-state checkpoint
    saves only. Positive values must be a multiple of `block_size`.

## Files

- `lmdeploy/pytorch/paging/block_trie.py`: trie match, allocation, eviction,
  stats, VLM extra hashes, SSM checkpoint ownership, and routed-expert replay.
- `lmdeploy/pytorch/paging/scheduler.py`: scheduler ownership and call order.
- `lmdeploy/pytorch/paging/state_manager.py`: runtime/checkpoint state-slot
  allocation.
- `lmdeploy/pytorch/paging/block_manager/*`: logical/physical block refs.
- `lmdeploy/pytorch/engine/inputs_maker.py`: block-table materialization and
  compact SSM state restore/save offsets.
- `lmdeploy/pytorch/engine/model_agent/agent.py`: stream-ordered SSM
  checkpoint restore/save copies around model forward.
- `lmdeploy/pytorch/engine/engine_loop.py`: checkpoint publish/release after
  forward input dispatch and output/event boundary.
- `lmdeploy/pytorch/messages.py`: `PrefixCacheState`, multimodal metas,
  `seq.cached_tokens`, and current-turn chunk-limit helpers.
- `lmdeploy/pytorch/multimodal/data_type.py`: stable multimodal content hashes.
- `lmdeploy/pytorch/backends/*/op_backend.py`: attention metadata that consumes
  `history_lengths`, `kv_seqlens`, and `block_offsets`.

## Flow

1. Scheduler creates `BlockTrie(cache_config, block_manager)`.
2. `Scheduler._schedule_prefill()` calls `block_trie.match(seq)` before
   eviction/allocation.
3. `BlockTrie.match()` is tentative: it can append shared logical KV blocks,
   advance `seq.num_history_ids`, set SSM restore metadata, replay routed
   experts, update stats, and call `seq.set_step(num_matched)` so model inputs
   contain only the uncached suffix.
4. If eviction, runtime SSM state admission, or prefill admission fails,
   scheduler rolls the tentative match back, restores stats, releases pinned
   checkpoint refs, frees temporary state/block refs, and clears
   `seq.cached_tokens`.
5. Scheduler allocates missing KV blocks with `block_manager.allocate(seq)` and,
   for SSM, runtime state with `state_manager.allocate(seq)`.
6. `block_trie.allocate(seq)` registers newly completed full blocks, owns one
   trie ref, replaces duplicate fresh blocks on collision, and frees duplicates.
7. `InputsMakerAsync.create_model_inputs*()` maps logical blocks to physical
   `block_offsets` and prepares compact state-copy offset pairs.
8. `model_forward()` restores matched SSM checkpoint state before forward and
   saves post-forward runtime state into reserved checkpoint slots after
   forward, using `StateCacheEngine.copy_caches()` on the forward stream.
9. `EngineLoop._publish_forward_prefix_cache()` commits reserved checkpoints
   and releases restore pins after the forward input is queued; save producer
   refs are released after the output/event boundary.

## Data Model

- `SchedulerSequence.history_cache` stores all token ids.
- Counters:
  - `num_history_ids`: computed prefix length for the next step.
  - `num_token_ids`: current query/suffix length.
  - `num_valid_ids`: `num_history_ids + num_token_ids`.
- `LogicalTokenBlocks` stores logical block ids and `last_shared_node`.
- Trie roots are partitioned by `adapter_name`; LoRA requests with different
  adapters must not share KV.
- Text/VLM trie lookup keys include token ids plus overlapping multimodal extra
  hashes. `PrefixCacheMeta(start, end, modality, content_hash)` records stable
  media identity and is lazily indexed into per-block `block_extra_hashes`.
- `MultiModalData.content_hash` is precomputed after model preprocessing when
  prefix caching is effectively enabled; the sequence path can fall back to
  `make_multimodal_content_hash()` for tests/defensive correctness.
- SSM trie nodes may own `state_idx`, `state_ready`, and `state_ref_count`.
  Ready checkpoints are indexed sparsely by `(adapter, step, last_block_hash)`;
  candidates are verified against ancestor token blocks and multimodal extras
  before reuse.
- Trie lookup uses Python hash plus exact token/extra-hash verification. The
  hash is only an in-process accelerator.
- `match()` uses `num_matched + block_size < seq.num_valid_ids`, intentionally
  leaving at least the final token/block to recompute.
- `allocate()` uses `num_matched + block_size <= seq.num_valid_ids`, registering
  newly filled full blocks for later reuse.
- `PrefixCacheState.match_recompute_blocks` asks matching to leave one or more
  full blocks out of the public hit. AR-spec/MTP sets it to 1 so target hidden
  bridge data can be recomputed.
- `private_recompute_start_step/private_recompute_end_step` mark trie-known
  overlap blocks that must remain private and writable during the following
  allocation instead of being substituted back to shared trie blocks.
- `seq.cached_tokens` is current-prompt cached-token accounting. It excludes
  previous-session history, recompute/preemption replay, rolled-back hits, and
  recompute-overlap blocks.

## Invariants

- Prefix cache owns one ref to every trie block; each sharing sequence owns an
  additional ref.
- `block_manager.free(seq)` releases sequence refs. `block_trie.evict()` may
  evict only leaf nodes whose ref count is `1`; leaf bookkeeping is an eviction
  index, not the source of truth.
- Never write new KV into a shared prefix block; matched sequences must advance
  with `seq.set_step(matched_tokens)` before model input creation.
- `WindowBlockManager` drops old blocks, so sliding-window reuse remains unsafe.
- SSM matches must have an exact ready checkpoint. If no checkpoint matches,
  treat it as a miss even if KV blocks match.
- SSM restore/save copies must be stream-ordered tensor operations in
  `model_forward()`; do not add CUDA synchronizes or GPU boolean masking in the
  model-agent hot path.
- Checkpoint restore/save refs pin slots across async scheduling windows so LRU
  eviction cannot reuse a source/destination slot before queued copies are safe.
- Matching must not start the suffix from inside a multimodal span. Use
  `SchedulerSequence.clamp_prefix_cache_match_step()` and include VLM extra
  hashes for every overlapping block.
- Long-context chunking must use current-turn multimodals by default. Include
  older history only when a real prefix-cache match recorded
  `match_start_step` and advanced past it via `get_chunk_limit_multimodals()`.
- Prefix hits may resume from the middle of a still-multi-chunk prompt. The
  scheduler/input-maker must preserve `match_start_step`, current-turn
  multimodal chunk limits, and accepted cached-token accounting when doing so.
- AR-spec/MTP hits must preserve the recompute-overlap window until
  `BlockTrie.allocate()` clears it. A rollback must clear the private recompute
  window and reset cached-token accounting.
- `block_size` is the scheduler/trie block size; `kernel_block_size` is the
  kernel page size. Split mappings must stay aligned with
  `InputsMakerAsync._map_to_kernel_block_offsets()`.

## Validation

For a hybrid model, use identical prompt IDs to compare a cold request, a request following unrelated traffic with no hit, and a confirmed hit. Record the actual cached-token count, the matched boundary, and the effective checkpoint/save alignment and state budget when the target version exposes them. A repeated short prompt can still miss. If greedy outputs diverge, find the first differing token and compare its competing scores before inspecting the restored KV and recurrent-state slots. A near tie can change the winner with a small score difference.

- Unit: `tests/pytorch/paging/test_block_trie.py` for match/allocate/evict ref
  counts, VLM extras, SSM checkpoints, stale checkpoint cleanup, and leaf
  eviction.
- Unit: `tests/pytorch/paging/test_scheduler.py` for prefill/decode scheduling,
  rollback, state admission, cached-token accounting, and metrics.
- Unit: `tests/pytorch/engine/test_inputs_maker.py` for long-context/VLM chunk
  boundary behavior and state-copy offset preparation.
- Unit: `tests/pytorch/engine/test_cache_engine.py` for state-cache copy
  helpers.
- Integration: repeated-prefix text, AR-spec/MTP prefix hits, VLM
  same/different media, SSM checkpoint hit/miss, routed-expert replay, and
  `prefix_cache_hit_rate` metrics.
- Negative coverage: verify PD roles, sliding-window, TurboMind VLM, and
  legacy VLM paths remain disabled unless explicitly redesigned.

## Watch Points

- `Scheduler._schedule_migration()` has prefix-cache logic in the migration
  path; inspect it carefully before extending prefix cache to PD migration,
  changing migration admission, or transferring state checkpoints.
- Prefix cache metrics are exposed via `Scheduler.schedule_metrics` as
  `prefix_cache_hit_rate`.
- Engine-side `seq.cached_tokens` now feeds OpenAI-compatible
  `usage.prompt_tokens_details.cached_tokens` through async engine and serving
  usage objects.
- Spec/MTP prefix caching is supported only through the current AR-spec overlap
  policy. Future changes still need careful draft state ownership, chunk flag
  handling, rollback semantics, and metrics rules.
