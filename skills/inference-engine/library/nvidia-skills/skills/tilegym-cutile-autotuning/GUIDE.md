> 语义适用性核对：缓存key需覆盖影响合法性与性能的shape、dtype、device/architecture、stride/alignment、layout和alias条件。FTZ/近似数学改变数值契约，不能为了速度默认启用；按任务精度容差验证。

> 本地核对：不采用“所有kernel必须autotune”“含原配置就不会变慢”“首次快就停止验证”“host wrapper不影响性能”的普遍结论。根据瓶颈选择维度，恢复原位输入，保留固定fallback并验证计时噪声、JIT/启动和wrapper成本。原文8/30配置预算是示例。

> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[NVIDIA/skills@25340619336e](https://github.com/NVIDIA/skills/blob/25340619336e89bcc40365c28bfe9e790430468e/skills/tilegym-cutile-autotuning/SKILL.md)，2026-10-31 核对。

# CuTile Autotuning

Add autotuning to CuTile kernels using the `exhaustive_search` API with tune-once/cache/direct-launch pattern.

## Instructions

Follow the decision tree to classify the kernel, design a search space, implement the tune-once/cache/launch pattern, and validate performance.

1. **Classify** — use the Decision Tree to determine search dimensions (occupancy-only vs full tile search)
2. **Design search space** — select the matching template from `references/kernel-type-templates.md`; prune to ≤ 30 configs in the final code via arch filters (directed exploration probes may temporarily exceed this — see Design Philosophy)
3. **Implement** — add `exhaustive_search` + cache + `ct.launch` following the Step-by-Step Workflow; handle in-place writes with split-buffer if needed
4. **Test** — run correctness with autotune enabled and with `DISABLE_AUTOTUNE=1`
5. **Validate** — A/B benchmark against fixed best-known config; see `references/search-strategies.md`
6. **Shrink** — prune dead-weight configs that never win, targeting ≤ 8 configs per architecture to minimize compilation cost (Step 10)

## Task Router — Jump to What You Need

| What are you trying to do? | Go to |
|---|---|
| Add autotune to a new kernel (most common) | Quick Reference below → Workflow: Adding Autotune → `references/kernel-type-templates.md` (pick by kernel type: T1=elementwise, T2=in-place, T3=matmul, T4=persistent, T5=FMHA, T6=FP8, T7=grouped GEMM, T8=varlen attention, T9=dual-GEMM fusion) |
| Debug: data corruption / wrong results after first run | Pitfall #1 (In-Place Kernel) |
| Debug: autotune taking 5+ minutes | Pitfall #2 (Compilation Timeout) |
| Debug: search space generator returning zero configs | Pitfall #5 first; also check arch filters, size guards, and `num_ctas` constraints |
| Optimize an existing autotune config | Workflow: Optimizing an Existing Config |

## Quick Reference — Occupancy-Only Autotune (Tune-Once/Cache/Launch)

Most CuTile kernels (elementwise, reduction, LayerNorm) need only occupancy tuning. Copy this pattern:

```python
from types import SimpleNamespace
from cuda.tile.tune import exhaustive_search
import cuda.tile as ct
import torch

def _my_autotune_configs():
    for occ in [1, 2, 4, 8]:
        yield SimpleNamespace(occupancy=occ)

# Module-level cache: tune once, launch fast forever after
_autotune_cache = {}

def my_op(x, output):
    stream = torch.cuda.current_stream()
    NUM_SM = torch.cuda.get_device_properties(x.device).multi_processor_count

    # Cache key: anything that affects optimal config (use str() for device)
    cache_key = (x.shape, x.dtype, str(x.device))

    if cache_key not in _autotune_cache:
        configs = list(_my_autotune_configs())
        result = exhaustive_search(
            configs,
            stream,
            grid_fn=lambda cfg: (min(NUM_SM * cfg.occupancy, M), 1, 1),
            kernel=my_kernel,
            args_fn=lambda cfg: (x, output, ...),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        best_cfg = result.best.config
        tuned_kernel = my_kernel.replace_hints(occupancy=best_cfg.occupancy)
        _autotune_cache[cache_key] = (best_cfg, tuned_kernel)  # cache BOTH

    cfg, tuned_kernel = _autotune_cache[cache_key]
    grid = (min(NUM_SM * cfg.occupancy, M), 1, 1)
    ct.launch(stream, grid, tuned_kernel, (x, output, ...))
```

Key rules:
- **Tune once, cache, launch directly** — `exhaustive_search` runs only on first call per shape; subsequent calls use cached config + `ct.launch`; measure lookup/dispatch overhead in the target workload
- For in-place kernels use split-buffer during search (separate input/output tensors)
- Keep ≤ 30 configs in final code (see Design Philosophy for temporary directed probes)
- `exhaustive_search` requires a `Sequence` (list/tuple) — convert generators with `list()`
- **Search space must include the original fixed config** — this guarantees autotuning never makes performance worse

**When to use this pattern**: Kernel has fixed block size (not tile-size tunable). Includes: elementwise (SwiGLU, GeGLU), reduction (RMSNorm, LayerNorm), RoPE, and persistent kernels with heuristic block sizes (grouped GEMM).

For complex kernels (matmul with tile sizes, FMHA, FP8 with num_ctas), read the full guide below + [`kernel-type-templates.md`](references/kernel-type-templates.md).

> **⚠️ Three pitfalls catch almost everyone — check before submitting:**
> - **`replace_hints` on hot path?** → Cache BOTH config AND kernel object from `exhaustive_search`. Calling `replace_hints()` every invocation recompiles (100–500× slower) → Pitfall #7
> - **In-place kernel** (writes back to input tensor)? → MUST use split-buffer pattern during search → Pitfall #1
> - **Search space empty?** → Check arch filters and `num_ctas` constraints → Pitfall #5

> **Minimum coverage**: On sm100+, FMHA/matmul/varlen search spaces must include both `num_ctas=1` and `num_ctas=2`. For core dimensions (tile sizes, occupancy), keep at least 2 distinct values even if unsure which is better — let `exhaustive_search` decide.

> **When to stop tuning**: A mean speedup in [0.98, 1.02] means your *current* search space isn't helping — but doesn't mean no config will help. Before stopping, check whether you've covered the key dimensions for this kernel type (consult `references/kernel-type-templates.md`). If the search space already covers the template's recommended dimensions and the best result is still noise-floor, then stop — further micro-adjustments won't help. If key dimensions are missing (e.g., never tried `num_ctas=2` for a dual-GEMM kernel), expand the search space rather than giving up.
>
> Once correctness tests pass and the autotuned kernel shows speedup over the fixed-config baseline, use the task-agreed repeat count and uncertainty criterion before claiming a gain. GPU kernel timing fluctuates ±5–10 % between invocations due to clock scaling and OS scheduling; a subsequent timing dip does not mean your code is wrong.
>
> To improve speedup, only modify the autotune search space (configs, tile sizes, occupancy, num_ctas). Do not modify other code (Python wrapper, stream management, etc.) to chase speedup — host-side dispatch, synchronization and stream management can affect end-to-end performance; keep changes within the requested scope.

## Reading Guide

- **Occupancy-only kernels** (elementwise, reduction, persistent with fixed block sizes): Quick Reference + Pitfall Checklist is sufficient — skip `references/` docs. For in-place kernels, also read Pitfall #1.
- **Complex kernels** (matmul with tunable tile sizes, FMHA, FP8 with num_ctas): Quick Reference → Decision Tree → API Reference → Step-by-Step Workflow → relevant `references/` docs.

**5-step summary**: Classify kernel → Design search space ([`parameter-space-design.md`](references/parameter-space-design.md)) → Implement using template ([`kernel-type-templates.md`](references/kernel-type-templates.md)) → Validate with A/B test → Check Pitfall Checklist.

**Reading references**: Read only the reference relevant to your kernel type — e.g., for FMHA, read the Template 5 section in `references/kernel-type-templates.md`; for hardware constraints, read only the target architecture's section. Avoid reading all references end-to-end when a targeted lookup suffices.

## Design Philosophy

**Build a small, precise search space bottom-up — not a large space trimmed down.** CuTile compilation is much heavier than Triton (~0.5-1s per config), so the **final code** should contain ≤ 30 configs. The approach is: classify the kernel type first, then construct only the relevant configs for that type and architecture.

**Directed exploration during development**: If the initial template configs yield speedup < 1.0, you may run a *temporary* larger probe (30–100 configs) via `bash + python3 -c` to identify which dimensions matter — but this probe must be **directional**, not a blind cartesian product. Use the kernel type classification to decide *which* dimensions to vary (e.g. for dual-GEMM, probe `num_ctas × occupancy` while fixing tile sizes; for FMHA, probe `TILE_M × num_ctas` while fixing TILE_N). Once the probe identifies the winning region, lock the final code's search space to ≤ 8 top candidates. Do NOT write the large probe into the source file — it is a one-shot diagnostic tool.

## Decision Tree: What Search Dimensions Does This Kernel Need?

All kernels should have autotuning added. The question is not *whether* to autotune, but *what dimensions* to search:

```
What type of kernel is this?
├── Compute-bound (matmul, GEMM, FMHA) → Does it have multiple tunable dimensions (tile sizes)?
│   ├── YES → Is it a fused multi-GEMM kernel (dual-GEMM, e.g. Linear+GLUAct)?
│   │   ├── YES → Template 9: low occupancy (1–2), conservative tiles (2× SHMEM/register pressure)
│   │   └── NO  → Full search: TILE_M × TILE_N × (TILE_K) × occupancy × num_ctas
│   │             (see matmul/FMHA templates in kernel-type-templates.md)
│   └── NO  → Occupancy-only search: [1, 2, 4, 8]
│             (see Quick Reference above)
├── Balanced (LayerNorm, reduction + compute) →
│   Occupancy-only search: [1, 2, 4, 8]
│   Expected benefit: 2-15%
└── Memory-bound (CE Loss, pure elementwise) →
    Occupancy-only search: [1, 2, 4, 8]
    Expected benefit: 0-15% (varies by kernel; zero-cost after tuning)
```

**Why memory-bound kernels only search occupancy (not num_ctas or tile sizes)**:
- **`num_ctas` has zero benefit**: `num_ctas > 1` enables TMA multicast, where multiple CTAs share tile data in shared memory (e.g., matmul A/B tiles reused across CTAs). Memory-bound kernels use per-element `ct.gather`/`ct.scatter` with no tile reuse — multi-CTA cooperation adds overhead with no data sharing benefit.
- **Tile sizes are pre-determined**: BLOCK_SIZE for memory-bound kernels is determined by offline sweep (e.g., 1024 is globally optimal on B200 across [256, 512, 1024, 2048, 4096, 8192]). This is a constant, not a runtime tunable.
- **Occupancy is the only effective knob**: Higher occupancy lets the GPU hide memory latency by switching to another CTA while one is stalled on a memory request.

> **Evidence — CE Loss experiment**: A 12-config search (occupancy × num_ctas) on Cross-Entropy Loss yielded only 2.5% gain (0.79x → 0.81x vs Triton). The `num_ctas` dimension contributed nothing; the result was reverted because compilation cost outweighed the marginal benefit. Occupancy-only (4 configs) achieves the same result at 3x less compilation time.

**Note on memory-bound kernels**: Consider occupancy-only autotune when expected savings amortize search and cache costs:
- The tune-once/cache/launch pattern avoids repeated search, but has lookup/dispatch costs
- The search space is tiny (4 configs, ~2-4s compilation)
- Even small improvements have value at scale

## Occupancy Selection Guide

Occupancy controls how many CTAs run concurrently per SM. Use this as a starting point when designing the occupancy search space:

| Occupancy Range | Best For | Example Kernels |
|-----------------|----------|-----------------|
| 1–4 | Compute-bound (heavy math) | Complex transforms, matmul |
| 4–8 | Balanced (GEMM, TMA) | Matrix multiply, FMHA |
| 8–16 | Memory-bound (reductions) | Softmax, LayerNorm |
| 16–32 | Very light (copies, casts) | Type conversions, elementwise |

Use these ranges to seed your initial search space. For occupancy-only kernels, `[1, 2, 4, 8]` covers most cases — see Quick Reference above.

## exhaustive_search API Reference

See [references/api-reference.md](references/api-reference.md) for the full
`exhaustive_search` API surface — current signature, `TuningResult`, the
tune-once/cache/launch pattern, `replace_hints`, kernel hints, `search_space`
design, and `grid_fn` patterns.

## Step-by-Step Workflow

See [references/workflow.md](references/workflow.md) for the end-to-end
workflow — adding autotune to a new kernel, handling existing
multi-architecture configs, integration with `torch.autograd.Function`,
cross-backend config transfer (Triton → CuTile), and optimizing an existing
config.

## Pitfall Checklist

See [references/pitfalls.md](references/pitfalls.md) for the full list of
common pitfalls — in-place data corruption, compilation timeout, cold-cache
performance skew, NCU profiling interference, `search_space` generator
exhaustion, FP8 precision loss, and `replace_hints` recompilation on hot
paths.

## Scope and Boundaries

This skill covers *only* autotune configuration: search space design, `exhaustive_search` invocation, caching, and `ct.launch` with tuned hints. It does **not** modify kernel code.

**In scope** (autotune config):
- Search space generator functions
- `exhaustive_search()` calls and result handling
- `kernel.replace_hints()` for applying tuned hints
- Cache logic (key design, dict management)
- `ct.launch()` with tuned kernel
- `DISABLE_AUTOTUNE` fallback path

**Out of scope** (kernel code modifications — do NOT make these changes):
- Math flags (flush_to_zero, rounding_mode)
- Performance Hints (slice_hint, buffer_depth, copy_config)
- Memory access patterns (2D→1D gather/scatter conversion)
- Codegen optimizations (safe_offs → padding_value)
- Algorithm changes (K-loop split, load balancing)

## Further Optimization Suggestions

After adding autotuning, the following kernel-level optimizations may yield additional gains. These are *outside the scope of this skill* — mention them to the user as potential next steps, but do not implement them as part of autotuning:

- **Math flags**: `flush_to_zero=True` + `rounding_mode=APPROX` can provide 34-72% improvement for FMHA-class kernels (set via environment variables `TILEIR_ENABLE_FTZ=1 TILEIR_ENABLE_APPROX=1` or in kernel code). *Causal chain*: larger tiles initially *decrease* performance by 18-43% due to subnormal handling overhead; enabling FTZ+APPROX rescues this and flips the result to +34-72%. Math flags are therefore a *prerequisite* for large-tile configs to be effective on FMHA-class kernels.
- **Performance Hints**: `slice_hint`, `buffer_depth`, `copy_config` — requires modifying kernel IR code
- **Memory access patterns**: Using TMA loads (`ct.load`) instead of `ct.gather`; removing unnecessary bounds checks (`check_bounds=False` when safe)
- **Codegen quality**: Using `padding_value` parameter instead of manual `ct.where` masking; removing `safe_offs`
- **Algorithm restructuring**: K-loop split, load balancing, algebraic simplification

## Differences from Triton Autotune

Key differences: Triton uses `@triton.autotune` decorator with `Config(...)` objects; CuTile uses `exhaustive_search()` with `SimpleNamespace` configs + separate cache + `ct.launch`. CuTile has no `num_warps`/`num_stages` (compiler decides) — only tile sizes + `occupancy` + `num_ctas`. CuTile compilation is heavier (keep ≤30 configs in final code). CuTile cache is user-managed in-memory (no automatic persistence). CuTile separates `args_fn` (kernel args) from `hints_fn` (compiler hints).

## Reference Documents

| Category | Document | Content |
|----------|----------|---------|
| **API Reference** | [`api-reference.md`](references/api-reference.md) | `exhaustive_search` signature, `TuningResult`, tune-once/cache/launch pattern, `replace_hints`, kernel hints, `search_space` design, `grid_fn` patterns |
| **Workflow** | [`workflow.md`](references/workflow.md) | End-to-end workflow: adding autotune to a new kernel, multi-architecture configs, `torch.autograd.Function` integration, Triton→CuTile transfer, optimizing existing configs |
| **Pitfalls** | [`pitfalls.md`](references/pitfalls.md) | Common pitfalls: in-place corruption, compilation timeout, cold-cache skew, NCU interference, `search_space` exhaustion, FP8 precision, `replace_hints` recompilation |
| **Parameter Design** | [`parameter-space-design.md`](references/parameter-space-design.md) | Per-kernel-type parameter spaces, cross-arch patterns, grid_fn patterns, pruning rules |
| **Search Strategies** | [`search-strategies.md`](references/search-strategies.md) | Exhaustive search, A/B test methodology, DISABLE_AUTOTUNE pattern |
| **Templates** | [`kernel-type-templates.md`](references/kernel-type-templates.md) | Copy-paste autotune templates for 8 kernel types |
| **Hardware** | [`hardware-constraints.md`](references/hardware-constraints.md) | Per-architecture constraints, tile size ranges, num_ctas rules, TMA requirements |

## Source Code References

Key files: `ops/cutile/matmul.py` (matmul autotune), `ops/cutile/attention.py` (FMHA autotune), `suites/unsloth/cutile/ct_ops.py` (shared `autotune_configs()` occupancy=[1,2,4,8]), `suites/unsloth/cutile/swiglu.py` (elementwise example), `suites/unsloth/cutile/rope_embedding.py` (split-buffer pattern), `suites/unsloth/cutile/grouped_gemm.py` (persistent GEMM, occupancy-only).

## Worked Examples

Each example shows the **before → after** pattern: `fixed_launch.py` (hardcoded `ct.launch`) and `autotuned_launch.py` (refactored to tune-once/cache/launch).

| Directory | Kernel | Autotune Pattern | Complexity | Key Teaching Point |
|-----------|--------|-----------------|------------|-------------------|
| [`assets/examples/01_rmsnorm_occupancy_only/`](https://github.com/NVIDIA/skills/blob/25340619336e89bcc40365c28bfe9e790430468e/skills/tilegym-cutile-autotuning/assets/examples/01_rmsnorm_occupancy_only) | RMSNorm (reduction) | Occupancy-only `[1,2,4,8]` | Low | Most common pattern — no tile tuning, just find best occupancy. Grid = `NUM_SM * cfg.occupancy`. Not in-place. |
| [`assets/examples/02_matmul_full_search/`](https://github.com/NVIDIA/skills/blob/25340619336e89bcc40365c28bfe9e790430468e/skills/tilegym-cutile-autotuning/assets/examples/02_matmul_full_search) | GEMM C=A@B | Full: `TILE_M/N/K` + `occupancy` + `num_ctas` (sm90+) | High | Compute-bound kernel with multiple tunable dimensions. `args_fn` passes tile sizes as `ct.Constant[int]`. `grid_fn` depends on `cfg`. ≤30 configs. |
| [`assets/examples/03_rope_inplace_splitbuffer/`](https://github.com/NVIDIA/skills/blob/25340619336e89bcc40365c28bfe9e790430468e/skills/tilegym-cutile-autotuning/assets/examples/03_rope_inplace_splitbuffer) | RoPE embedding (in-place) | Occupancy-only, with split-buffer | Medium | In-place kernel MUST use split-buffer during search to avoid corruption. Search writes to scratch; final `ct.launch` uses real in-place args. |
