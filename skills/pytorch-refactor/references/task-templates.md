# Common Task Templates

## Template 1: Add a New Model Architecture Rewrite

1. Compare the target config, weight layout, and forward contract with the nearest existing model rewrite.
2. If that implementation fits, reuse it and add the architecture name to `module_map.py` only when resolution needs a new entry. Implement the differing behavior in `models/` when the contracts differ.
3. Add `DEVICE_SPECIAL_MODULE_MAP` entries only for device-specific behavior.
4. Verify which class the worker builds and sample a representative loaded tensor against the checkpoint, including the packed or sharded layout when applicable. In a local LMDeploy 0.18.0 build, supplying `custom_module_map` in the engine config alone left a new architecture unresolved. In the single-process PyTorch pipeline, `update_custom_module_map()` in the model-building process registered the rewrite. The worker reads `ModelConfig.custom_module_map`, so a populated `MiscConfig` by itself did not demonstrate handoff. Check the registration path in the target checkout, especially with worker processes.
5. For first bring-up, use fixed input IDs and a short greedy generation through the normal pipeline. Compare the prefill prediction and at least one decode step with the HF implementation on those IDs. For hybrid state or different cache layouts, also verify continuation using the actual slot and block mapping.

## Template 1b: Add a VLM / Omni Multimodal Model

1. Trace architecture detection through `archs.py::check_vl_llm()` and `models/module_map.py`. Add a `pytorch/configurations/*` builder when the existing path cannot unwrap the HF config.
2. Check the closest registered VL processor against the target's preprocessing, special tokens, and transformer-version requirements. Reuse it when those contracts match. Register a new processor in `vl/model/builder.py` when the target needs its own `VisionModel.build_preprocessor()`.
3. Wire public request parsing for any new modality in
   `serve/processors/multimodal.py` and `vl/media/*`.
4. Extend `VisionModel.ATTR_NAME_TO_MODALITY`, `FEATURE_NAMES`, and
   `vl/model/preprocess_utils.py::get_expanded_mm_items()` for processor
   outputs, bundled-item splitting, offsets, and prompt-order restoration.
5. Reuse or adapt the PyTorch model and `BaseModelInputProcessor` for the target's inputs.
   Preserve `MultiModalData.start/end`, `modality`, `mrope_pos_ids`, and
   modality token ids in `meta`.
6. Update `DeployModelMixinV1.get_multimodal_mask()` if the model scatters
   embeddings by modality token id.
7. Validate with processor unit tests covering mixed modalities, offsets,
   token counts, MROPE, and compact tensor storage, plus one model-build smoke
   when feasible.

## Template 2: Refactor Model Agent Forward Flow

1. Locate the existing owner of the data-shaping step. Keep the logic there when it is already clear. Extract a helper when callers share the operation or a distinct shape transformation needs independent verification.
2. Keep `model_forward()` context-management order unchanged
3. Keep sampling input/output structures backward-compatible
4. Exercise the phase whose behavior changes. When both phases share the changed path, exercise prefill, decode, and the handoff between them.

## Template 2b: Add a New Field to ModelInputs

Adding a new tensor/scalar field that must survive across decode steps requires changes in multiple locations. Follow this checklist:

### 1. Declare the field
- `lmdeploy/pytorch/model_inputs.py` — add field to `ModelInputs` dataclass

### 2. Populate during prefill
- `lmdeploy/pytorch/engine/inputs_maker.py` — set field in `create_model_inputs()` or `create_model_inputs_long_context()`
- OR in `strategies/<paradigm>/model_agent.py` — via `make_extra_inputs()` if paradigm-specific

### 3. Carry through prefill→decode transition
- `lmdeploy/pytorch/strategies/ar/model_inputs.py` — `get_model_inputs_next_decoding()`
- `lmdeploy/pytorch/strategies/dllm/step_inputs.py` — `_get_model_inputs_next_decoding()`
- `lmdeploy/pytorch/strategies/ar_spec/step_inputs.py` — uses AR conversion, then may expand spec-token fields

### 4. Carry through merge (prefill batch → decode batch)
- `lmdeploy/pytorch/strategies/ar/model_inputs.py` — `merge_model_inputs()`: add `torch.cat` or similar
- This function is shared by AR, DLLM, and AR_Spec `StepInputs`

### 5. Carry through reindex (eviction / invalid sequence removal)
- `lmdeploy/pytorch/strategies/ar/model_inputs.py` — `index_select_model_inputs()`
- `lmdeploy/pytorch/strategies/dllm/step_inputs.py` — `_reindex_model_inputs_dllm()`
- `lmdeploy/pytorch/strategies/ar_spec/step_inputs.py` — `_reindex_model_inputs_arspec()`

### 6. Handle in-place step advance (if needed)
- `lmdeploy/pytorch/model_inputs.py` — `ModelInputs.step()`: update field if it changes each decode step

### 7. Populate in dummy inputs (if needed for CUDA graph warmup)
- `lmdeploy/pytorch/strategies/base/model_inputs.py` — `make_dummy_inputs()`: add field

### Current pain points
- Prefill-to-decode and reindex logic is split across AR, DLLM, and AR_Spec, so fields are easily forgotten.
- DLLM helpers do not currently carry `mrope_pos_ids`.
- AR_Spec reindex handles `mrope_pos_ids` but does not carry
  `target_hidden_states`, `target_position_ids`, or `target_inputs_embeds`;
  verify whether a new field is target-only or persistent decode state.

### Field classification for index_select
| Category | Index pattern | Examples |
|----------|--------------|---------|
| Batch-dim-0 | `val[indices]` | `seq_length`, `history_lengths`, `local_adapter_ids`, `state_offsets` |
| Token-dim | `val[..., indices]` (after reshape for DLLM/AR_Spec) | `input_ids` |
| Special-dim | `val[:, indices]` (dim=1) | `mrope_pos_ids` |
| Spec/draft | shape-specific; inspect producer/consumer | `target_hidden_states`, `target_position_ids`, `target_inputs_embeds` |
| Aggregate | pass through or override from delta | `max_q_seqlen`, `max_kv_seqlen`, `sum_kv_seqlen` |
| List | `[val[i] for i in indice_cpu]` | `model_metas` |
| Non-indexable | dropped (set to None) | `vision_inputs`, `dp_meta`, `is_dummy` |

## Template 3: Adjust Executor Selection Logic

1. Apply policy changes only in `engine/executor/__init__.py`
2. Preserve `dp > 1` to `ray` unless requirements explicitly change it
3. Validate both `world_size=1` and multi-GPU build paths

## Template 4: Change Prefix Caching or Long-Context Chunking

1. Classify the change as prefix reuse, long-context chunking, scheduling policy,
   or attention/cache metadata. Prefix cache and long-context chunking are
   separate features unless the requested design explicitly connects them.
2. Read `references/prefix-caching.md` for prefix-cache work or
   `references/long-context-chunking.md` for chunking work before editing.
3. Update config wiring first if adding public knobs:
   `lmdeploy/messages.py` -> `engine/config_builder.py` -> `config.py` -> CLI/server paths.
4. Preserve scheduler order:
   `block_trie.match()` -> eviction -> `block_manager.allocate()` -> `block_trie.allocate()`.
5. Preserve sequence counters:
   matched or completed chunks must move into `num_history_ids` before model input creation.
6. Keep prefix caching scoped to the supported paths: AR text, AR-spec/MTP with
   recompute overlap, PyTorch VLM extra hashes, and SSM checkpoints. Treat
   sliding-window, PD/disaggregated, TurboMind VLM, and legacy PyTorch VLM as
   unsupported unless intentionally redesigned and tested.
7. If chunk flags or `ModelInputs` fields change, update AR, AR_Spec, DLLM, spec decode, and dummy inputs as needed.
8. Match the check to the failure. For a cache-key change, verify the expected trie hit and miss. For admission or chunk boundaries, exercise the scheduler or input maker with the triggering sequence. If the change crosses into model execution, verify the affected transition through the normal engine entry point.

## Template 5: Change DisAgg / PD Migration

1. Read `references/disagg-pd.md` and `references/cache-engine.md` block-layout
   sections.
2. Keep role/config wiring aligned:
   `lmdeploy/messages.py` -> `engine/config_builder.py` -> `config.py` ->
   `disagg/config.py` consumers.
3. Preserve the `block_size == kernel_block_size` PD restriction unless the
   migration offset/stride math is explicitly redesigned.
4. Validate `engine_checker.py` rejects unsupported layouts before runtime.
5. Test both engine-side P2P methods and proxy-side connection orchestration
   when changing protocol, backend, or endpoint metadata.

## Recommended Refactor Order

Start at the component that owns the changed value or state. For a public configuration change, follow it from `lmdeploy/messages.py` through validation and dispatch to the consuming backend. For a local fix, follow the affected callers and consumers from the owning component.

## Validation by Change

| Changed behavior | Evidence to obtain |
| --- | --- |
| Pure configuration, shape, or cache-key logic | Exercise the triggering input and boundary through the owning component. |
| Model rewrite or device build | Instantiate an impacted architecture or build configuration. For forward behavior, run the affected phase. |
| State carried from prefill to decode | Observe that state before and after the transition in the same request. |
| Executor selection or rank coordination | Check `world_size=1` when that branch is affected. Exercise the multi-rank topology for distributed behavior. |

Use existing coverage for these checks. Add a test when the user or repository requires it, or when a specific regression would otherwise go undetected.
