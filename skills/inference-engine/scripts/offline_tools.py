#!/usr/bin/env python3
"""Stable CPU-only entrypoints for the bundled inference analysis tools."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TOOLS = {
    'profile': 'skills/llm-torch-profiler-analysis/scripts/analyze_llm_torch_profile.py',
    'capacity': 'skills/llm-serving-capacity-planner/scripts/capacity_analyzer.py',
    'compare': 'skills/llm-serving-auto-benchmark/scripts/compare_benchmark_results.py',
    'plan': 'skills/llm-serving-auto-benchmark/scripts/validate_cookbook_configs.py',
    'pipeline': 'skills/llm-pipeline-analysis/scripts/layer_timeline_analyzer.py',
    'layer-kernels': 'skills/llm-pipeline-analysis/scripts/layer_kernel_breakdown.py',
    'time-map': 'skills/llm-pipeline-analysis/scripts/perfetto_time_mapper.py',
    'compute': 'skills/model-compute-simulation/scripts/model_compute_simulator.py',
    'compute-trace': 'skills/model-compute-simulation/scripts/extract_compute_flow_from_trace.py',
    'layer-track': 'skills/torch-profiler-layer-track/scripts/add_layer_track.py',
    'compact': 'skills/torch-profiler-layer-track/scripts/compact_gpu_tracks.py',
    'review-query': 'skills/sglang-humanize-review/scripts/query_sglang_review_corpus.py',
    'review-summary': 'skills/sglang-humanize-review/scripts/summarize_sglang_review_corpus.py',
    'model-history': 'model-pr-optimization-history/scripts/query.py',
    'day0-check': 'skills/model-optimization/sglang-model-day0-support/scripts/validate_day0_bundle.py',
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library-root', type=Path,
                        default=Path(os.environ.get('INFERENCE_SKILL_LIBRARY') or ROOT / 'library'),
                        help='Library directory containing bbuf; defaults to INFERENCE_SKILL_LIBRARY or this bundle/library.')
    parser.add_argument('--list', action='store_true', help='List offline tools and script availability.')
    parser.add_argument('tool', nargs='?', choices=TOOLS)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    bbuf = args.library_root.expanduser().resolve() / 'bbuf'
    if args.list:
        for name, path in TOOLS.items():
            status = 'available' if (bbuf / path).is_file() else 'not-installed'
            print(f'{name}\t{status}\t{path}')
        return 0
    if not args.tool:
        parser.print_help()
        return 0
    forwarded = args.arguments
    if forwarded[:1] == ['--']:
        forwarded = forwarded[1:]
    options = {a.split('=', 1)[0] for a in forwarded if a.startswith('--')}
    live_options = {'--url', '--mapping-url', '--formal-url', '--host', '--port'}
    if any('url' in option or any(live.startswith(option) for live in live_options) for option in options):
        parser.error('This entrypoint accepts offline files only; live capture/replay requires a separate authorized workflow.')
    help_only = '-h' in forwarded or '--help' in forwarded
    if args.tool == 'profile' and not help_only:
        if '--input' not in options and not {'--mapping-input', '--formal-input'} <= options:
            parser.error('profile requires --input or both --mapping-input and --formal-input.')
    script = bbuf / TOOLS[args.tool]
    if not script.is_file():
        parser.error(f'Missing tool script: {script}. Restore this tool with its supporting files or select a complete --library-root; no automatic download is performed.')
    if args.tool in {'review-query', 'review-summary'} and '--corpus' not in options and not help_only:
        corpus = bbuf / 'skills/sglang-humanize-review/references/sglang-review-corpus.jsonl.gz'
        if not corpus.is_file():
            parser.error('Historical corpus is not installed here. Select a library containing it with --library-root, or pass a trusted --corpus file.')
    if args.tool == 'model-history' and not help_only:
        if not any((bbuf / 'model-pr-optimization-history' / name).is_dir()
                   for name in ('vllm', 'sglang', 'tensorrt_llm', 'tokenspeed')):
            parser.error('Model histories are not installed here. Select a library containing the required history with --library-root.')
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['NVIDIA_VISIBLE_DEVICES'] = 'void'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return subprocess.run([sys.executable, str(script), *forwarded], env=env, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
