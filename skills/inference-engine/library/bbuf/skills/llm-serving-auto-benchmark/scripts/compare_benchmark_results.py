#!/usr/bin/env python3
"""Summarize normalized cross-framework benchmark JSONL results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


def _get(row: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = row
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _float(row: dict[str, Any], path: str, default: float = 0.0) -> float:
    value = _get(row, path, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(row: dict[str, Any], path: str, default: bool = False) -> bool:
    value = _get(row, path, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _p50_or_mean_ttft_ms(row: dict[str, Any]) -> float:
    if _get(row, "metrics.p50_ttft_ms") is not None:
        return _float(row, "metrics.p50_ttft_ms", 1e30)
    return _float(row, "metrics.mean_ttft_ms", 1e30)


def _p50_or_mean_tpot_ms(row: dict[str, Any]) -> float:
    if _get(row, "metrics.p50_tpot_ms") is not None:
        return _float(row, "metrics.p50_tpot_ms", 1e30)
    return _float(row, "metrics.mean_tpot_ms", 1e30)


def _rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _get(row, "status") == "ok",
        _bool(row, "sla.passed"),
        _float(row, "metrics.request_throughput"),
        _float(row, "metrics.output_token_throughput"),
        -_p50_or_mean_ttft_ms(row),
        -_p50_or_mean_tpot_ms(row),
        -_float(row, "hardware.gpu_count", 1e30),
    )


def _finite_metric(row: dict[str, Any], path: str) -> float | None:
    value = _get(row, path)
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sla_limits(row: dict[str, Any]) -> dict[str, float | None]:
    sla = _get(row, "sla")
    if not isinstance(sla, dict):
        return {}
    return {key: _finite_metric(row, "sla." + key)
            for key, value in sla.items() if key.startswith(("min_", "max_")) and value is not None}


def _validation_issues(row: dict[str, Any]) -> list[str]:
    issues = []
    if _get(row, "status") != "ok" or not _bool(row, "sla.passed"):
        issues.append("run status or declared SLA did not pass")
    for path in ("framework", "model", "hardware.gpu_model"):
        value = _get(row, path)
        if not isinstance(value, str) or value.strip().lower() in {"", "unknown", "n/a"}:
            issues.append(path + " is missing")
    gpu_count = _finite_metric(row, "hardware.gpu_count")
    if gpu_count is None or gpu_count <= 0 or not gpu_count.is_integer():
        issues.append("hardware.gpu_count is missing or invalid")
    workload = row.get("workload")
    labels = {"scenario", "scenario_name", "dataset_scenario"}
    if not isinstance(workload, dict) or not any(
        value is not None and value != "" for key, value in workload.items() if key not in labels
    ):
        issues.append("workload identity or load parameters are missing")
    rate = _finite_metric(row, "metrics.success_rate")
    minimum = _finite_metric(row, "sla.min_success_rate")
    if rate is None or not 0 <= rate <= 1:
        issues.append("observed success_rate is missing or invalid")
    if minimum is None or not 0 <= minimum <= 1:
        issues.append("min_success_rate is missing or invalid; do not infer the SLA")
    elif rate is not None and rate < minimum:
        issues.append("observed success_rate fails the SLA")
    failed = _finite_metric(row, "metrics.failed")
    if failed is None:
        failed = _finite_metric(row, "metrics.failed_requests")
    if failed is not None:
        if failed < 0 or failed != int(failed):
            issues.append("failed request count is invalid")
        elif failed > 0 and rate == 1:
            issues.append("failed requests contradict success_rate=1")
    for metric in ("request_throughput", "output_token_throughput"):
        value = _finite_metric(row, "metrics." + metric)
        if value is None or value < 0:
            issues.append(metric + " is missing, negative, or non-finite")
    for key, limit in _sla_limits(row).items():
        if key == "min_success_rate":
            continue
        if not re.fullmatch(r"max_p(?:100|[1-9]?[0-9])_(?:ttft|tpot|e2e)_ms", key):
            issues.append("cannot verify SLA criterion " + key)
            continue
        metric = key.removeprefix("max_")
        value = _finite_metric(row, "metrics." + metric)
        if limit is None or limit < 0 or value is None or value < 0 or value > limit:
            issues.append("observed " + metric + " does not establish its SLA")
    normalization = _get(row, "normalization", {})
    if isinstance(normalization, dict) and any(value is False for value in normalization.values()):
        issues.append("normalization reports a comparison mismatch")
    return issues


def _is_winner_candidate(row: dict[str, Any]) -> bool:
    return not _validation_issues(row)


def _incomparable_scenarios(rows: list[dict[str, Any]]) -> set[str]:
    cohorts: dict[str, set[str]] = {}
    for row in rows:
        # Conservative grouping: tuning across different load/resource conditions
        # is a separate analysis and must not become an automatic engine ranking.
        contract = {
            "model": row.get("model"),
            "model_revision": row.get("model_revision"),
            "tokenizer_revision": row.get("tokenizer_revision"),
            "workload": row.get("workload"),
            "gpu_model": _get(row, "hardware.gpu_model"),
            "gpu_count": _finite_metric(row, "hardware.gpu_count"),
            "sla_limits": _sla_limits(row),
        }
        signature = json.dumps(contract, sort_keys=True, ensure_ascii=False)
        cohorts.setdefault(_scenario(row), set()).add(signature)
    return {scenario for scenario, signatures in cohorts.items() if len(signatures) > 1}


def _ranking_issues(row: dict[str, Any], incomparable: set[str]) -> list[str]:
    issues = _validation_issues(row)
    if _scenario(row) in incomparable:
        issues.append("comparison metadata differs within scenario")
    return issues


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _cell(value: Any, digits: int = 2) -> str:
    text = _fmt(value, digits)
    return text.replace("\n", "<br>").replace("|", "\\|")


def _scenario(row: dict[str, Any]) -> str:
    for path in (
        "workload.scenario",
        "workload.scenario_name",
        "workload.dataset_scenario",
        "workload.dataset_name",
        "workload.kind",
        "scenario",
    ):
        value = _get(row, path)
        if value:
            return str(value)
    return "default"


def _server_command(row: dict[str, Any]) -> str:
    return str(_get(row, "server_command") or _get(row, "launch_command") or "")


def _artifact_summary(row: dict[str, Any]) -> str:
    artifacts = _get(row, "artifacts", {})
    if not isinstance(artifacts, dict):
        return ""
    parts = []
    for key in ("raw_result", "server_log", "benchmark_log", "summary"):
        value = artifacts.get(key)
        if value:
            parts.append(f"{key}: {value}")
    return "<br>".join(parts)


def _optional_metric(row: dict[str, Any], *paths: str) -> Any:
    for path in paths:
        value = _get(row, path)
        if value is not None:
            return value
    return None


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{line_no}: expected a JSON object")
            rows.append(row)
    return rows


def best_by_framework_and_scenario(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    incomparable = _incomparable_scenarios(rows)
    for row in rows:
        if _scenario(row) in incomparable or not _is_winner_candidate(row):
            continue
        key = (str(_get(row, "framework", "unknown")), _scenario(row))
        if key not in best or _rank_key(row) > _rank_key(best[key]):
            best[key] = row
    return sorted(
        best.values(), key=lambda row: (_scenario(row), _rank_key(row)), reverse=True
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    incomparable = _incomparable_scenarios(rows)
    fields = [
        "framework",
        "scenario",
        "candidate_id",
        "status",
        "sla_passed",
        "ranking_eligible",
        "ranking_evidence",
        "request_throughput",
        "output_token_throughput",
        "p50_ttft_ms",
        "p50_tpot_ms",
        "mean_ttft_ms",
        "mean_tpot_ms",
        "p99_ttft_ms",
        "p99_tpot_ms",
        "mean_accept_length",
        "pre_scheduler_ms",
        "prefix_cache_hit_rate",
        "gpu_count",
        "server_command",
        "failure_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            issues = _ranking_issues(row, incomparable)
            writer.writerow(
                {
                    "framework": _get(row, "framework", ""),
                    "scenario": _scenario(row),
                    "candidate_id": _get(row, "candidate_id", ""),
                    "status": _get(row, "status", ""),
                    "sla_passed": _bool(row, "sla.passed"),
                    "ranking_eligible": not issues,
                    "ranking_evidence": "; ".join(issues),
                    "request_throughput": _get(row, "metrics.request_throughput", ""),
                    "output_token_throughput": _get(
                        row, "metrics.output_token_throughput", ""
                    ),
                    "p50_ttft_ms": _get(row, "metrics.p50_ttft_ms", ""),
                    "p50_tpot_ms": _get(row, "metrics.p50_tpot_ms", ""),
                    "mean_ttft_ms": _get(row, "metrics.mean_ttft_ms", ""),
                    "mean_tpot_ms": _get(row, "metrics.mean_tpot_ms", ""),
                    "p99_ttft_ms": _get(row, "metrics.p99_ttft_ms", ""),
                    "p99_tpot_ms": _get(row, "metrics.p99_tpot_ms", ""),
                    "mean_accept_length": _optional_metric(
                        row,
                        "spec_decode.mean_accept_length",
                        "metrics.mean_accept_length",
                    ),
                    "pre_scheduler_ms": _optional_metric(
                        row,
                        "phase_metrics.pre_scheduler_ms",
                        "metrics.pre_scheduler_ms",
                    ),
                    "prefix_cache_hit_rate": _optional_metric(
                        row,
                        "cache.prefix_cache_hit_rate",
                        "metrics.prefix_cache_hit_rate",
                    ),
                    "gpu_count": _get(row, "hardware.gpu_count", ""),
                    "server_command": _server_command(row),
                    "failure_reason": _get(row, "failure_reason", ""),
                }
            )


def _append_best_commands_by_framework(
    lines: list[str], scenario_winners: list[dict[str, Any]]
) -> None:
    frameworks = sorted(
        {str(_get(row, "framework", "unknown")) for row in scenario_winners}
    )
    lines.extend(["## Best Commands By Framework", ""])
    for framework in frameworks:
        lines.extend(
            [
                f"### `{framework}`",
                "",
                "| Scenario | Candidate | Status | SLA | Req/s | Output tok/s | Total tok/s | P50 TTFT ms | P50 TPOT ms | Success rate | Accept len | Pre-scheduler ms | Cache hit | GPUs | Server command | Artifacts |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        rows = [row for row in scenario_winners if _get(row, "framework") == framework]
        for row in sorted(rows, key=_scenario):
            lines.append(
                "| {scenario} | {candidate} | {status} | {sla} | {rps} | {otps} | {ttps} | {ttft} | {tpot} | {success} | {accept} | {sched} | {cache} | {gpus} | {command} | {artifacts} |".format(
                    scenario=_cell(_scenario(row)),
                    candidate=_cell(_get(row, "candidate_id", "")),
                    status=_cell(_get(row, "status", "")),
                    sla=_cell(_bool(row, "sla.passed")),
                    rps=_cell(_get(row, "metrics.request_throughput")),
                    otps=_cell(_get(row, "metrics.output_token_throughput")),
                    ttps=_cell(_get(row, "metrics.total_token_throughput")),
                    ttft=_cell(_get(row, "metrics.p50_ttft_ms")),
                    tpot=_cell(_get(row, "metrics.p50_tpot_ms")),
                    success=_cell(_get(row, "metrics.success_rate")),
                    accept=_cell(
                        _optional_metric(
                            row,
                            "spec_decode.mean_accept_length",
                            "metrics.mean_accept_length",
                        )
                    ),
                    sched=_cell(
                        _optional_metric(
                            row,
                            "phase_metrics.pre_scheduler_ms",
                            "metrics.pre_scheduler_ms",
                        )
                    ),
                    cache=_cell(
                        _optional_metric(
                            row,
                            "cache.prefix_cache_hit_rate",
                            "metrics.prefix_cache_hit_rate",
                        )
                    ),
                    gpus=_cell(_get(row, "hardware.gpu_count")),
                    command=_cell(_server_command(row)),
                    artifacts=_cell(_artifact_summary(row)),
                )
            )
        lines.append("")


def _append_cross_framework_table(
    lines: list[str], scenario_winners: list[dict[str, Any]]
) -> None:
    lines.extend(
        [
            "## Cross-Framework Best Comparison",
            "",
            "| Scenario | Rank | Framework | Candidate | SLA | Req/s | Output tok/s | P50 TTFT ms | P50 TPOT ms | Accept len | Pre-scheduler ms | Cache hit | GPUs | Server command |",
            "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    scenario_names = sorted({_scenario(row) for row in scenario_winners})
    for scenario_name in scenario_names:
        rows = [row for row in scenario_winners if _scenario(row) == scenario_name]
        for rank, row in enumerate(sorted(rows, key=_rank_key, reverse=True), 1):
            lines.append(
                "| {scenario} | {rank} | {framework} | {candidate} | {sla} | {rps} | {otps} | {ttft} | {tpot} | {accept} | {sched} | {cache} | {gpus} | {command} |".format(
                    scenario=_cell(scenario_name),
                    rank=rank,
                    framework=_cell(_get(row, "framework", "")),
                    candidate=_cell(_get(row, "candidate_id", "")),
                    sla=_cell(_bool(row, "sla.passed")),
                    rps=_cell(_get(row, "metrics.request_throughput")),
                    otps=_cell(_get(row, "metrics.output_token_throughput")),
                    ttft=_cell(_get(row, "metrics.p50_ttft_ms")),
                    tpot=_cell(_get(row, "metrics.p50_tpot_ms")),
                    accept=_cell(
                        _optional_metric(
                            row,
                            "spec_decode.mean_accept_length",
                            "metrics.mean_accept_length",
                        )
                    ),
                    sched=_cell(
                        _optional_metric(
                            row,
                            "phase_metrics.pre_scheduler_ms",
                            "metrics.pre_scheduler_ms",
                        )
                    ),
                    cache=_cell(
                        _optional_metric(
                            row,
                            "cache.prefix_cache_hit_rate",
                            "metrics.prefix_cache_hit_rate",
                        )
                    ),
                    gpus=_cell(_get(row, "hardware.gpu_count")),
                    command=_cell(_server_command(row)),
                )
            )
    lines.append("")


def render_markdown(rows: list[dict[str, Any]]) -> str:
    scenario_winners = best_by_framework_and_scenario(rows)

    lines = ["# Benchmark Summary", ""]
    if not rows:
        lines.append("No rows found.")
        return "\n".join(lines) + "\n"

    incomparable = _incomparable_scenarios(rows)
    lines.extend([
        "Automatic ranking checks observed metrics and recorded comparison metadata; "
        "it does not prove model quality, data identity, or artifact authenticity.",
        "",
    ])
    if incomparable:
        lines.extend(["Automatic ranking withheld for mismatched workload/model/hardware/SLA scenarios: "
                      + ", ".join(sorted(incomparable)), ""])
    _append_best_commands_by_framework(lines, scenario_winners)
    _append_cross_framework_table(lines, scenario_winners)
    lines.extend(["## All Recorded Candidates", "",
                  "| Framework | Scenario | Candidate | Success rate | Failed requests | Ranking evidence |",
                  "| --- | --- | --- | ---: | ---: | --- |"])
    for row in rows:
        issues = _ranking_issues(row, incomparable)
        lines.append("| " + " | ".join(_cell(value) for value in [
            _get(row, "framework"), _scenario(row), _get(row, "candidate_id"),
            _get(row, "metrics.success_rate"),
            _get(row, "metrics.failed", _get(row, "metrics.failed_requests")),
            "; ".join(issues) if issues else "eligible within recorded cohort",
        ]) + " |")
    lines.append("")

    failed = [
        row
        for row in rows
        if not _is_winner_candidate(row)
    ]
    if failed:
        lines.extend(
            [
                "",
                "## Candidates With Failed Or Incomplete Evidence",
                "",
                "These candidates are excluded by the evidence checks. Reported status and SLA are original input fields; the exclusion reason determines eligibility.",
                "",
                "| Framework | Candidate | Reported status | Reported SLA | Eligibility | Reason |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in failed:
            reasons = _validation_issues(row)
            reported_reason = _get(row, "failure_reason", "")
            if reported_reason:
                reasons.append(str(reported_reason))
            lines.append(
                "| {framework} | {candidate} | {status} | {sla} | excluded | {reason} |".format(
                    framework=_cell(_get(row, "framework", "")),
                    candidate=_cell(_get(row, "candidate_id", "")),
                    status=_cell(_get(row, "status", "")),
                    sla=_cell(_bool(row, "sla.passed")),
                    reason=_cell("; ".join(reasons)),
                )
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="Normalized JSONL")
    parser.add_argument("--output", required=True, type=Path, help="Markdown summary")
    parser.add_argument("--csv", type=Path, help="Optional CSV table")
    args = parser.parse_args()

    rows = load_rows(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(rows), encoding="utf-8")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.csv, sorted(rows, key=_rank_key, reverse=True))


if __name__ == "__main__":
    main()
