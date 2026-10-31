> 来源参考，按本任务需要读取。工程专属的安装、运行时、资源池、宿主插件、重试预算和对外操作不自动成为本工作台要求；运行相关命令前核对目标 checkout 和已授权范围。完整工程依赖见本专题及仓库索引。

> 来源：[mit-han-lab/KernelWiki@76d27b56f804](https://github.com/mit-han-lab/KernelWiki/blob/76d27b56f804e7e7295d4c570e1e5d7eef4b0a75/SKILL.md)，2026-10-31 核对。

# KernelWiki — Blackwell & Hopper Kernel Optimization Wiki

> **Knowledge cutoff: 2026-04-27.** All upstream PR data, blog summaries, and version-claim entries reflect upstream state on or before this date (per `data/refresh-cutoff.yaml`). Re-run the refresh tooling to advance the cutoff.

Query a structured, cross-referenced knowledge base of GPU kernel optimization for NVIDIA Blackwell (SM100) and Hopper (SM90) — 2179 merged PRs, 48 wiki synthesis pages, 7 competitions, 20 blogs, 11 doc summaries.

## When To Use This Skill

Trigger this skill when the user asks about:

- **Blackwell/SM100 kernel programming** — tcgen05.mma, TMEM, CLC, 2-SM cooperative, NVFP4, FP8/FP4 block scaling, PDL/GDC
- **Kernel implementations** — FlashAttention-4, DeepGEMM, FlashMLA, NSA, GatedDeltaNet, NVFP4 GEMM/GEMV, fused MoE, gated dual GEMM
- **Performance patterns** — low SM utilization, memory-bound, register pressure, compute-bound, tail effects, pipeline stalls
- **DSLs for Blackwell** — CuTe DSL, CUDA C++ with PTX inline, Triton on Blackwell
- **Hopper → Blackwell migration** — wgmma → tcgen05, register → TMEM accumulators
- **PR references** — "how did vLLM/SGLang/FlashInfer/CUTLASS/PyTorch implement X for SM100?"
- **Competition solutions** — GPU Mode NVFP4 hackathon, FlashInfer MLSys 2026 submissions

Do NOT use this skill for:

- Generic CUDA questions unrelated to Blackwell/Hopper tensor cores
- Host-side framework integration (model loading, request routing, scheduling policy)
- Distributed systems topics — DeepEP, EPLB, DualPipe are out of scope

## How To Query

All commands below run from the skill directory (the clone root — the directory this `SKILL.md` lives in). The scripts auto-resolve the wiki root; **no environment variable required**.

### Path 1: Unified search (preferred for natural language)

```bash
python3 scripts/query.py "how to fuse gate-up dual GEMM on Blackwell"
python3 scripts/query.py --tag nvfp4 --type kernel
python3 scripts/query.py --repo cutlass --limit 20
python3 scripts/query.py --symptom tail-effect --compact
```

Filters: `--type`, `--tag`, `--repo`, `--language`, `--architecture`,
`--symptom`, `--confidence`, `--limit`, `--compact`, `--paths-only`. `--tag`
and `--architecture` accept aliases — `--tag UMMA` matches `tcgen05`,
`--architecture B200` matches `sm100`, etc.

### Path 2: Fetch a specific page by id or path

```bash
python3 scripts/get_page.py kernel-flash-attention-4
python3 scripts/get_page.py pr-cutlass-2472
python3 scripts/get_page.py kernel-flash-attention-4 --follow-sources
python3 scripts/get_page.py kernel-flash-attention-4 --body-only
```

### Path 3: Regex text search across wiki bodies and PR pages

```bash
python3 scripts/grep_wiki.py "tcgen05\\.fence"
python3 scripts/grep_wiki.py "2-CTA backward" --only wiki
python3 scripts/grep_wiki.py "nvfp4" "block_scale" --any
```

### Path 4: Pre-built cross-reference indices

Auto-generated under `queries/`:

- `queries/by-problem.md` — symptom → pattern page → candidate techniques
- `queries/by-technique.md` — 15 techniques with architectures, confidence, reproducibility, source count
- `queries/by-hardware-feature.md` — tcgen05/tmem/clc/tma/nvfp4/etc. → related wiki + PR pages
- `queries/by-kernel-type.md` — gemm/attention/moe/mla/gated-delta-net → pages
- `queries/by-language.md` — cute-dsl/cuda-cpp/ptx/triton → guide page + related kernels/sources
- `queries/by-repo.md` — all 2179 PRs across cutlass/sglang/vllm/flashinfer/pytorch/DeepGEMM

### Path 5: Primer, schema, examples

Companion docs under `references/`:

- `references/primer.md` — topic map: hardware features, techniques, symptoms, canonical page IDs. Read this first when the question is broad.
- `references/schema.md` — condensed frontmatter schema, confidence rules, reproducibility ladder, controlled vocabulary, canonical aliases.
- `references/examples.md` — 10 worked query patterns mapping user questions → command sequences → synthesis.

## Output Pattern

When answering from this KB:

1. **Cite specific pages** with paths (e.g., `wiki/kernels/flash-attention-4.md`) and IDs (`kernel-flash-attention-4`).
2. **Follow `sources:` fields** to trace claims back to PRs/blogs/docs.
3. **Respect confidence levels** — `verified` > `source-reported` > `inferred` > `experimental`. Call out when a claim is `experimental` or `inferred`.
4. Read each page's semantic corrections first. `snippet` records the presence of a fragment, not compilation or numerical verification; do not execute fragments identified as invalid. Follow the pinned upstream implementation and the workbench technical contracts.
5. **Report performance claims with all six fields** — `gpu`, `dtype`, `shape`, `metric`, `value`, `source_id`.

## Knowledge Base Contents (knowledge cutoff: 2026-04-27)

- **2265 total markdown pages** — 2179 PR references + 48 wiki synthesis + 20 blogs + 11 docs + 7 contests
- **6 candidate ledgers** in `candidates/` — 4,222 merged PRs classified (include/defer/exclude) Jan 2025 – Apr 2026
- **89 verbatim/extracted/derived asset bundles** in `artifacts/` (PR diffs, kernel files, blog code) — pinned to upstream SHAs via `PROVENANCE.yaml`
- **6 auto-generated query indices** in `queries/`
- **Controlled vocabulary** (80+ tags) in `data/tags.yaml`, alias map in `data/aliases.yaml`
- **Hybrid version-claim registry** — per-page `version_sensitive: <id>` pointers + `data/version-claims.yaml` central registry, validated for bidirectional consistency
- **Validator** `scripts/validate.py` — 2265 files / 89 bundles / 6 ledgers / 0 errors
- **Blackwell-first** — SM90 pages only appear when they carry explicit `blackwell_relevance`

The knowledge cutoff date is the last day on which upstream PRs / blog snapshots were refreshed. To advance it: run `scripts/refresh_candidate_ledger.py`, regenerate PR pages, then bump `data/refresh-cutoff.yaml::cutoff_date`.

## Quality Guarantees

- `verified` checks evidence metadata, not the correctness of every statement or implementation.
- The validator checks snippet presence and metadata; compilability, math, synchronization, and actual GPU behavior require separate evidence.
- Every PR page has `inclusion_reason` and `status: merged`
- All Hopper-inclusive pages have explicit `blackwell_relevance`
