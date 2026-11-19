# PyTorch Linear Dispatch

Use this reference when changing `lmdeploy/pytorch/nn/linear`, linear backend builders, or model code that constructs linear projections.

## Dispatch Structure

1. Start at the model's builder call in `lmdeploy/pytorch/nn/linear/__init__.py`. Trace the projection prefix and the ordinary, merged, or QKV builder it uses. When `build_linear` receives a quantization config, it reads the active build context's `quant_config.get_quant_method(prefix, module_kind="linear")`. An ignored projection or an FP8 `moe_only` scope can select `BaseLinear`.
2. Inspect the constructed projection and its implementation with `type(layer)` and `type(layer.impl)`. On checkouts with static FP8 support, `fp8` with static activations and no weight block size selects `StaticF8Linear` in `build_linear`. Other FP8 configurations there select `BlockedF8Linear`. AWQ, smooth quant, and static W8A8 have their own wrappers.
3. Follow the backend call in that wrapper. Typed backends pass a spec such as `LinearStaticF8BuildSpec` to `get_backend().build_op(...)`. In `backends/cuda/op_backend.py`, the spec selects an implementation or a provider builder. If a wrapper in an older revision calls `get_layer_impl_builder`, follow the returned builder there.
4. Read the selected implementation's `forward` for the actual local shape, dtype, device and effective flags. For example, CUDA W4A16's `_build_linear_w4a16` in `backends/cuda/awq_modules.py` checks provider support during construction. An unsupported `auto` choice uses `AwqLinearW4A16Impl`. Explicit `turbomind` raises with the support reason. Static FP8 also makes a per-call choice inside the same implementation class.

`LinearBase` owns TP, DP_TP and LoRA forward composition. The quantized wrapper owns its weight layout and loader.

## Verify the selected path

Run the relevant command from the checkout with the Python environment used by the engine. Check `lmdeploy.__file__` if a wheel is also installed.

- For FP8 quantization scope, `python -m pytest -q tests/pytorch/config/test_fp8_quant_scope.py::test_quantization_config_fp8_moe_only_module_kinds` checks whether linear and MoE select different methods. Inspect the resulting wrapper for the actual projection prefix.
- For static FP8 CUDA math, `python -m pytest -q tests/pytorch/kernel/test_static_fp8_linear.py::test_static_fp8_scaled_mm_matches_triton` compares the scaled-matmul and Triton paths at the shapes in that test. It constructs the backend directly. To check model wiring, call the projection built by the model-facing builder with the same weights, scales, layout and input.
- For a token-count fallback, record the calls reached by that projection using a trace or counters around the real functions. In `backends/cuda/static_fp8_modules.py`, with scaled matmul and compiled quantization enabled, E4M3 weights and a token allowlist of `[1]`, one token selects compiled quantization. Three tokens select `per_tensor_quant_fp8`. Both calls still use `torch._scaled_mm`. Compare each output to the reference at that shape. For a different backend or revision, select a test that calls its observed implementation.

## Boundaries To Preserve

- Keep model code using builder functions instead of direct wrapper construction.
- Prefer construction-time dispatch over per-forward branching unless runtime state truly changes.
- Treat TP/DP_TP communication, LoRA composition, quantized weight loading, and backend kernel selection as separate concerns.
- Check backend capability differences explicitly, especially default/cuda/dlinfer behavior around `all_reduce`, `reduce_scatter`, and blocked FP8 support.

## Future Cleanup Opportunities

- Preserve the active config, projection prefix and module kind when consolidating repeated quantization lookup.
- Consider a small quantization family registry where each family provides adapters for normal, merged, and QKV builders; do not force all quantization classes through one broad kwargs path.
- Add focused tests for factory selection, TP shard loading, DP_TP gather/reduce paths, LoRA with all-reduce, and backend fallback behavior before larger refactors.
