#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# jetson-llm-benchmark: benchmark an Ollama model via its /api/generate REST
# endpoint. No benchmark container required — the Ollama daemon must be
# reachable at --endpoint (native install or a container exposing that port).
# Timing data (prompt_eval_duration, eval_duration, etc.) comes from Ollama's
# own non-streaming response; no --verbose parsing needed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILLS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=/dev/null
. "${SKILLS_ROOT}/jetson-diagnostic/scripts/detect_jetson.sh"

usage() {
  cat <<'EOF'
Usage: bench_ollama.sh --model <name> [options]
  --model <name>       Ollama model name already available to the daemon
  --num-prompts <n>    Measured requests, default: 20
  --input-len <n>      Approximate prompt token count, default: 512
  --output-len <n>     Max tokens to generate per request, default: 128
  --endpoint <url>     Ollama base URL, default: http://localhost:11434
  --no-warmup          Skip the 1-request warmup pass
EOF
}

MODEL=""
NP=20
ILEN=512
OLEN=128
ENDPOINT="http://localhost:11434"
WARMUP=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)       MODEL="$2"; shift 2;;
    --num-prompts) NP="$2"; shift 2;;
    --input-len)   ILEN="$2"; shift 2;;
    --output-len)  OLEN="$2"; shift 2;;
    --endpoint)    ENDPOINT="$2"; shift 2;;
    --no-warmup)   WARMUP=0; shift;;
    -h|--help)     usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

[[ -z "${MODEL}" ]] && { echo "ERROR: --model is required" >&2; usage; exit 2; }

# --- Preflight ---
if ! curl -sf "${ENDPOINT}/api/tags" >/dev/null 2>&1; then
  if command -v ollama >/dev/null 2>&1; then
    echo "ERROR: Ollama is installed but the daemon is not running. Start it with: ollama serve" >&2
  else
    echo "ERROR: Ollama daemon not reachable at ${ENDPOINT}." >&2
    echo "  Install: curl -fsSL https://ollama.com/install.sh | sh" >&2
    echo "  Or if running in a container, check --endpoint matches the exposed port." >&2
  fi
  exit 3
fi
curl -sf "${ENDPOINT}/api/tags" \
  | MODEL="${MODEL}" python3 -c '
import os, sys, json
d = json.load(sys.stdin)
model = os.environ["MODEL"]
names = [m["name"] for m in d.get("models", [])]
if model not in names:
    available = ", ".join(names)
    sys.stderr.write("ERROR: model " + repr(model) + " not found. Available: " + available + "\n")
    sys.exit(1)
' || exit 3

WARN=()
if command -v nvpmodel >/dev/null 2>&1; then
  NVPMODEL_OUT="$(nvpmodel -q 2>/dev/null || true)"
  NVPMODEL_NAME="$(printf '%s\n' "${NVPMODEL_OUT}" | awk -F': ' '/NV Power Mode/ {print $2; exit}')"
  NVPMODEL_ID="$(printf '%s\n' "${NVPMODEL_OUT}" | awk '/NV Power Mode/ {getline; print $1; exit}')"
  case "${NVPMODEL_NAME}" in
    MAXN|MAXN_*) ;;
    *) WARN+=("nvpmodel is ${NVPMODEL_NAME:-unknown} (id ${NVPMODEL_ID:-unknown}); MAXN/MAXN_* preferred for comparable results") ;;
  esac
fi

# --- Build request body (written to a temp file to avoid shell-escaping issues) ---
REQ_FILE="$(mktemp)"
MODEL="${MODEL}" ILEN="${ILEN}" OLEN="${OLEN}" python3 - "${REQ_FILE}" <<'PY'
import json, math, os, random, sys

n_words = max(1, math.ceil(int(os.environ["ILEN"]) / 1.3))
words   = ["the","quick","brown","fox","jumps","over","lazy","dog","a","an",
           "of","in","to","and","is","was","that","it","he","she","they","with"]
prompt  = " ".join(random.choices(words, k=n_words))

with open(sys.argv[1], "w") as f:
    json.dump({
        "model":   os.environ["MODEL"],
        "prompt":  prompt,
        "stream":  False,
        "options": {"num_predict": int(os.environ["OLEN"])},
    }, f)
PY

do_request() {
  curl -sf "${ENDPOINT}/api/generate" \
    -H 'Content-Type: application/json' \
    --data-binary "@${REQ_FILE}"
}

parse_timing() {
  python3 -c "
import sys, json
d = json.load(sys.stdin)
def ms(key):
    value = d.get(key)
    return value / 1e6 if isinstance(value, (int, float)) and value >= 0 else None
prompt_ms = ms('prompt_eval_duration')
eval_ms = ms('eval_duration')
load_ms = ms('load_duration')
total_ms = ms('total_duration')
eval_count = d.get('eval_count')
valid_count = isinstance(eval_count, (int, float)) and eval_count > 0
per_token_ms = eval_ms / eval_count if eval_ms is not None and valid_count else None
tok_s = eval_count / (eval_ms / 1000) if eval_ms is not None and eval_ms > 0 and valid_count else None
without_load_ms = total_ms - load_ms if total_ms is not None and load_ms is not None else None
print(json.dumps({'prompt_eval_ms': prompt_ms, 'eval_ms_per_token': per_token_ms,
                  'eval_tok_s': tok_s, 'server_total_ms': total_ms,
                  'server_load_ms': load_ms, 'server_without_load_ms': without_load_ms}))
"
}

# --- Warmup (result discarded) ---
if [[ "${WARMUP}" -eq 1 ]]; then
  do_request | parse_timing >/dev/null 2>&1 || true
fi

# --- Measured runs: one JSON object per line ---
RESULTS_FILE="$(mktemp)"
for (( i=0; i<NP; i++ )); do
  do_request | parse_timing >> "${RESULTS_FILE}"
done

# --- Aggregate p50 / p99 / mean ---
WARN_JSON="$(printf '%s\n' "${WARN[@]+"${WARN[@]}"}" \
  | python3 -c 'import sys,json; print(json.dumps([l.rstrip() for l in sys.stdin if l.strip()]))')"

BENCH_MODEL="${MODEL}" \
BENCH_WARN="${WARN_JSON}" \
BENCH_ILEN="${ILEN}" \
BENCH_OLEN="${OLEN}" \
BENCH_NP="${NP}" \
JETSON_SKU="${JETSON_SKU:-unknown}" \
JETSON_GENERATION="${JETSON_GENERATION:-unknown}" \
JETSON_PRODUCT_LINE="${JETSON_PRODUCT_LINE:-unknown}" \
JETSON_VARIANT="${JETSON_VARIANT:-unknown}" \
JETSON_L4T_VERSION="${JETSON_L4T_VERSION:-unknown}" \
python3 - "${RESULTS_FILE}" <<'PY'
import json, os, sys

rows = []
with open(sys.argv[1]) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

def pct(vals, p):
    vals = [v for v in vals if v is not None]
    if not vals: return None
    s = sorted(vals)
    idx = max(0, min(int(len(s) * p / 100), len(s) - 1))
    return round(s[idx], 2)

def mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else None

prompts = [r['prompt_eval_ms'] for r in rows]
evals = [r['eval_ms_per_token'] for r in rows]
rates = [r['eval_tok_s'] for r in rows]
totals = [r['server_total_ms'] for r in rows]
loads = [r['server_load_ms'] for r in rows]
without_load = [r['server_without_load_ms'] for r in rows]

out = {
    "skill":        "jetson-llm-benchmark",
    "runtime":      "ollama",
    "model":        os.environ["BENCH_MODEL"],
    "sku":          os.environ["JETSON_SKU"],
    "generation":   os.environ["JETSON_GENERATION"],
    "product_line": os.environ["JETSON_PRODUCT_LINE"],
    "variant":      os.environ["JETSON_VARIANT"],
    "l4t":          os.environ["JETSON_L4T_VERSION"],
    "container":    "native/ollama",
    "config": {
        "input_len":   int(os.environ["BENCH_ILEN"]),
        "output_len":  int(os.environ["BENCH_OLEN"]),
        "num_prompts": int(os.environ["BENCH_NP"]),
        "concurrency": 1,
    },
    "metrics": {
        "ttft_ms_p50": None,
        "ttft_ms_p99": None,
        "itl_ms_p50": None,
        "itl_ms_p99": None,
        "tpot_ms_p50": None,
        "throughput_tok_s": None,
        "e2e_latency_ms_p50": None,
        "server_prompt_eval_ms_p50": pct(prompts, 50),
        "server_generation_ms_per_token_p50": pct(evals, 50),
        "server_generation_tok_s_mean": mean(rates),
        "server_total_duration_ms_p50": pct(totals, 50),
        "server_load_duration_ms_p50": pct(loads, 50),
        "server_duration_excluding_load_ms_p50": pct(without_load, 50),
    },
    "metric_scope": "non-streaming Ollama server durations; client TTFT/ITL/E2E unmeasured",
    "warnings": json.loads(os.environ["BENCH_WARN"]),
}
print(json.dumps(out, indent=2))
PY

rm -f "${REQ_FILE}" "${RESULTS_FILE}"
