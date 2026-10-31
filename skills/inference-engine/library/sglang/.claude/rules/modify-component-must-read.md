# Must-Read Skills Before Modifying Components

Before modifying the following components, read the listed skill first.

- **Speculative decoding code** (anything under `python/sglang/srt/speculative/`, related attention backends, scheduler accumulators, IPC fields, observability metrics, or CLI flags) → [`speculative-naming`](../../.agents/skills/speculative-naming/GUIDE.md)
- **`Scheduler` / `TokenizerManager` / `ModelRunner` `__init__`** (`python/sglang/srt/managers/scheduler.py`, `python/sglang/srt/managers/tokenizer_manager.py`, `python/sglang/srt/model_executor/model_runner.py`) → [`large-class-style`](../../.agents/skills/large-class-style/GUIDE.md)
- **Any edit to a frozen core file** (currently `python/sglang/srt/model_executor/model_runner.py`) → [`large-class-style`](../../.agents/skills/large-class-style/GUIDE.md)
- **Environment variables** (adding, renaming, or reviewing any `SGLANG_*` env var, migrating a legacy `SGL_*` alias, or touching `python/sglang/srt/environ.py`) → [`env-var-conventions`](../../.agents/skills/env-var-conventions/GUIDE.md)
- **Chat rendering in `rust/sglang-processor`** (adding or changing a model under `rust/sglang-processor/src/render/`, or its parity fixtures) → [`processor-model-parity`](../../.agents/skills/processor-model-parity/GUIDE.md)
- **Scripted runtime** (anything related to the scripted runtime) → [`scripted-runtime-notes`](../../.agents/skills/scripted-runtime-notes/GUIDE.md)
