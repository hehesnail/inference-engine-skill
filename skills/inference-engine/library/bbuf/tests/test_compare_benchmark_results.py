from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "llm-serving-auto-benchmark"
    / "scripts"
    / "compare_benchmark_results.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("compare_benchmark_results", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CompareBenchmarkResultsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = load_module()

    def test_scenario_winner_prefers_successful_sla_passing_rows(self) -> None:
        rows = [
            {
                "framework": "sglang",
                "candidate_id": "sglang-fast-fail",
                "status": "ok",
                "workload": {"scenario": "chat", "dataset_name": "fixed-chat"},
                "sla": {"passed": False, "min_success_rate": 1.0},
                "metrics": {
                    "success_rate": 1.0,
                    "request_throughput": 99,
                    "output_token_throughput": 990,
                    "mean_ttft_ms": 1,
                    "mean_tpot_ms": 1,
                    "p99_ttft_ms": 1,
                    "p99_tpot_ms": 1,
                },
                "hardware": {"gpu_count": 1, "gpu_model": "H200"},
                "model": "fixture-model",
            },
            {
                "framework": "sglang",
                "candidate_id": "sglang-startup-fail",
                "status": "failed",
                "failure_reason": "server did not become ready",
                "workload": {"scenario": "startup"},
                "sla": {"passed": False, "min_success_rate": 1.0},
                "metrics": {},
                "hardware": {"gpu_count": 1, "gpu_model": "H200"},
                "model": "fixture-model",
            },
            {
                "framework": "sglang",
                "candidate_id": "sglang-steady",
                "status": "ok",
                "workload": {"scenario": "chat", "dataset_name": "fixed-chat"},
                "sla": {"passed": True, "min_success_rate": 1.0},
                "metrics": {
                    "success_rate": 1.0,
                    "request_throughput": 10,
                    "output_token_throughput": 100,
                    "mean_ttft_ms": 50,
                    "mean_tpot_ms": 5,
                    "p99_ttft_ms": 50,
                    "p99_tpot_ms": 5,
                },
                "hardware": {"gpu_count": 1, "gpu_model": "H200"},
                "model": "fixture-model",
            },
            {
                "framework": "vllm",
                "candidate_id": "vllm-best",
                "status": "ok",
                "workload": {"scenario": "chat", "dataset_name": "fixed-chat"},
                "sla": {"passed": True, "min_success_rate": 1.0},
                "metrics": {
                    "success_rate": 1.0,
                    "request_throughput": 12,
                    "output_token_throughput": 90,
                    "mean_ttft_ms": 60,
                    "mean_tpot_ms": 6,
                    "p99_ttft_ms": 60,
                    "p99_tpot_ms": 6,
                },
                "hardware": {"gpu_count": 1, "gpu_model": "H200"},
                "model": "fixture-model",
            },
        ]

        winners = self.mod.best_by_framework_and_scenario(rows)
        self.assertEqual(
            [row["candidate_id"] for row in winners],
            ["vllm-best", "sglang-steady"],
        )

        summary = self.mod.render_markdown(rows)
        self.assertIn("sglang-fast-fail", summary)
        self.assertIn("sglang-startup-fail", summary)
        self.assertIn("Best Commands By Framework", summary)
        self.assertIn("Cross-Framework Best Comparison", summary)
        self.assertNotIn("Overall Winner", summary)
        self.assertNotIn("Best Per Framework", summary)
        self.assertNotIn("| startup |", summary.split("## All Recorded Candidates")[0])
        self.assertIn("Candidates With Failed Or Incomplete Evidence", summary)

    def test_comparison_requires_identity_and_observed_sla(self):
        import csv
        from copy import deepcopy

        row = {"framework": "vllm", "model": "m", "candidate_id": "a", "status": "ok",
               "workload": {"scenario": "chat", "num_prompts": 16},
               "hardware": {"gpu_model": "H200", "gpu_count": 1},
               "sla": {"passed": True, "min_success_rate": 1, "max_p99_ttft_ms": 300},
               "metrics": {"request_throughput": 2, "output_token_throughput": 10,
                           "success_rate": 1, "p99_ttft_ms": 300, "mean_ttft_ms": 49}}
        other = deepcopy(row)
        other.update(framework="sglang", candidate_id="b")
        other["sla"]["max_p99_ttft_ms"] = "300"
        best = self.mod.best_by_framework_and_scenario
        self.assertEqual(len(best([row, other])), 2)
        self.assertNotIn("| 49 |", self.mod.render_markdown([row]))
        other["metrics"]["p99_ttft_ms"] = 301
        self.assertEqual(best([row, other]), [row])
        del other["metrics"]["p99_ttft_ms"]
        self.assertEqual(best([row, other]), [row])
        other["sla"]["max_p99_ttft_ms"] = 400
        self.assertEqual(best([row, other]), [])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            self.mod.write_csv(path, [row, other])
            with path.open() as handle:
                exported = list(csv.DictReader(handle))
        self.assertEqual([r["ranking_eligible"] for r in exported], ["False", "False"])
        self.assertIn("comparison metadata differs", exported[0]["ranking_evidence"])
        for key in ("model", "hardware", "workload"):
            partial = {k: v for k, v in row.items() if k != key}
            self.assertEqual(best([partial, partial]), [])
            self.assertIn("All Recorded Candidates", self.mod.render_markdown([partial]))

    def test_load_rows_rejects_non_object_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text("[1, 2, 3]\n", encoding="utf-8")

            with self.assertRaises(SystemExit):
                self.mod.load_rows(path)

    def test_writes_csv_with_failed_reason(self) -> None:
        rows = [
            {
                "framework": "trtllm",
                "candidate_id": "trt-c1",
                "status": "failed",
                "failure_reason": "server exited",
                "sla": {"passed": False, "min_success_rate": 1.0},
                "metrics": {},
                "hardware": {"gpu_count": 1, "gpu_model": "H200"},
                "model": "fixture-model",
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            self.mod.write_csv(path, rows)
            text = path.read_text(encoding="utf-8")

        self.assertIn("trt-c1", text)
        self.assertIn("server exited", text)
        self.assertIn("mean_ttft_ms", text)
        self.assertIn("mean_tpot_ms", text)
        self.assertNotIn("mmlu_accuracy", text)
        self.assertNotIn("gsm8k_accuracy", text)

    def test_failed_candidate_table_escapes_markdown_cells(self) -> None:
        rows = [
            {
                "framework": "trt|llm",
                "candidate_id": "trt|bad",
                "status": "failed|startup",
                "failure_reason": "server | exited",
                "sla": {"passed": False, "min_success_rate": 1.0},
                "metrics": {},
                "hardware": {"gpu_count": 1, "gpu_model": "H200"},
                "model": "fixture-model",
            }
        ]

        summary = self.mod.render_markdown(rows)

        self.assertIn("trt\\|llm", summary)
        self.assertIn("trt\\|bad", summary)
        self.assertIn("failed\\|startup", summary)
        self.assertIn("server \\| exited", summary)

    def test_renders_scenario_tables(self) -> None:
        rows = [
            {
                "framework": "sglang",
                "candidate_id": "sglang-c1",
                "status": "ok",
                "workload": {"scenario": "chat", "dataset_name": "fixed-chat"},
                "sla": {"passed": True, "min_success_rate": 1.0},
                "metrics": {
                    "request_throughput": 10,
                    "output_token_throughput": 100,
                    "mean_ttft_ms": 80,
                    "mean_tpot_ms": 6,
                    "p99_ttft_ms": 80,
                    "p99_tpot_ms": 6,
                    "success_rate": 1.0,
                },
                "hardware": {"gpu_count": 1, "gpu_model": "H200"},
                "model": "fixture-model",
                "server_command": "python -m sglang.launch_server --model-path m",
            },
            {
                "framework": "vllm",
                "candidate_id": "vllm-c1",
                "status": "ok",
                "workload": {"scenario": "chat", "dataset_name": "fixed-chat"},
                "sla": {"passed": True, "min_success_rate": 1.0},
                "metrics": {
                    "success_rate": 1.0,
                    "request_throughput": 12,
                    "output_token_throughput": 90,
                    "mean_ttft_ms": 85,
                    "mean_tpot_ms": 7,
                    "p99_ttft_ms": 85,
                    "p99_tpot_ms": 7,
                },
                "hardware": {"gpu_count": 1, "gpu_model": "H200"},
                "model": "fixture-model",
                "server_command": "vllm serve m",
            },
            {
                "framework": "sglang",
                "candidate_id": "sglang-c2",
                "status": "ok",
                "workload": {"scenario": "summarization", "dataset_name": "fixed-summary"},
                "sla": {"passed": True, "min_success_rate": 1.0},
                "metrics": {
                    "success_rate": 1.0,
                    "request_throughput": 8,
                    "output_token_throughput": 140,
                    "mean_ttft_ms": 120,
                    "mean_tpot_ms": 8,
                    "p99_ttft_ms": 120,
                    "p99_tpot_ms": 8,
                },
                "hardware": {"gpu_count": 1, "gpu_model": "H200"},
                "model": "fixture-model",
                "server_command": "python -m sglang.launch_server --model-path m --long",
            },
        ]

        scenario_winners = self.mod.best_by_framework_and_scenario(rows)
        self.assertEqual(
            {
                (row["framework"], row["workload"]["scenario"])
                for row in scenario_winners
            },
            {("sglang", "chat"), ("vllm", "chat"), ("sglang", "summarization")},
        )

        summary = self.mod.render_markdown(rows)
        self.assertIn("### `sglang`", summary)
        self.assertIn("| chat | sglang-c1", summary)
        self.assertIn("| summarization | sglang-c2", summary)
        self.assertNotIn("Overall Winner", summary)
        self.assertNotIn("Best Per Framework", summary)
        self.assertNotIn("Accuracy Of Selected Deployment Commands", summary)
        self.assertNotIn("MMLU", summary)
        self.assertNotIn("GSM8K", summary)
        self.assertIn("P50 TTFT ms", summary)
        self.assertIn("P50 TPOT ms", summary)
        self.assertNotIn("Mean TTFT ms", summary)
        self.assertNotIn("Mean TPOT ms", summary)
        self.assertNotIn("P99 TTFT ms", summary)
        self.assertNotIn("P99 TPOT ms", summary)

    def test_ranking_prefers_lower_mean_ttft_over_lower_p99_ttft(self) -> None:
        rows = [
            {
                "framework": "sglang",
                "candidate_id": "sglang-lower-mean",
                "status": "ok",
                "workload": {"scenario": "chat", "dataset_name": "fixed-chat"},
                "sla": {"passed": True, "min_success_rate": 1.0},
                "metrics": {
                    "success_rate": 1.0,
                    "request_throughput": 10,
                    "output_token_throughput": 100,
                    "mean_ttft_ms": 40,
                    "mean_tpot_ms": 5,
                    "p99_ttft_ms": 200,
                    "p99_tpot_ms": 5,
                },
                "hardware": {"gpu_count": 1, "gpu_model": "H200"},
                "model": "fixture-model",
            },
            {
                "framework": "sglang",
                "candidate_id": "sglang-lower-p99",
                "status": "ok",
                "workload": {"scenario": "chat", "dataset_name": "fixed-chat"},
                "sla": {"passed": True, "min_success_rate": 1.0},
                "metrics": {
                    "success_rate": 1.0,
                    "request_throughput": 10,
                    "output_token_throughput": 100,
                    "mean_ttft_ms": 50,
                    "mean_tpot_ms": 5,
                    "p99_ttft_ms": 100,
                    "p99_tpot_ms": 5,
                },
                "hardware": {"gpu_count": 1, "gpu_model": "H200"},
                "model": "fixture-model",
            },
        ]

        winners = self.mod.best_by_framework_and_scenario(rows)
        self.assertEqual([row["candidate_id"] for row in winners], ["sglang-lower-mean"])

    def test_ranking_prefers_lower_mean_tpot_over_lower_p99_tpot(self) -> None:
        rows = [
            {
                "framework": "sglang",
                "candidate_id": "sglang-lower-mean-tpot",
                "status": "ok",
                "workload": {"scenario": "chat", "dataset_name": "fixed-chat"},
                "sla": {"passed": True, "min_success_rate": 1.0},
                "metrics": {
                    "success_rate": 1.0,
                    "request_throughput": 10,
                    "output_token_throughput": 100,
                    "mean_ttft_ms": 40,
                    "mean_tpot_ms": 5,
                    "p99_ttft_ms": 100,
                    "p99_tpot_ms": 20,
                },
                "hardware": {"gpu_count": 1, "gpu_model": "H200"},
                "model": "fixture-model",
            },
            {
                "framework": "sglang",
                "candidate_id": "sglang-lower-p99-tpot",
                "status": "ok",
                "workload": {"scenario": "chat", "dataset_name": "fixed-chat"},
                "sla": {"passed": True, "min_success_rate": 1.0},
                "metrics": {
                    "success_rate": 1.0,
                    "request_throughput": 10,
                    "output_token_throughput": 100,
                    "mean_ttft_ms": 40,
                    "mean_tpot_ms": 6,
                    "p99_ttft_ms": 100,
                    "p99_tpot_ms": 10,
                },
                "hardware": {"gpu_count": 1, "gpu_model": "H200"},
                "model": "fixture-model",
            },
        ]

        winners = self.mod.best_by_framework_and_scenario(rows)
        self.assertEqual(
            [row["candidate_id"] for row in winners], ["sglang-lower-mean-tpot"]
        )

    def test_cli_writes_markdown_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "rows.jsonl"
            output_path = root / "summary.md"
            csv_path = root / "summary.csv"
            row = {
                "framework": "sglang",
                "candidate_id": "candidate-1",
                "status": "ok",
                "sla": {"passed": True, "min_success_rate": 1.0},
                "metrics": {
                    "success_rate": 1.0,
                    "request_throughput": 1.5,
                    "output_token_throughput": 8.0,
                    "mean_ttft_ms": 10.0,
                    "mean_tpot_ms": 2.0,
                },
                "hardware": {"gpu_count": 1, "gpu_model": "H200"},
                "model": "fixture-model",
            }
            input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            rows = self.mod.load_rows(input_path)
            output_path.write_text(self.mod.render_markdown(rows), encoding="utf-8")
            self.mod.write_csv(csv_path, rows)

            self.assertIn("candidate-1", output_path.read_text(encoding="utf-8"))
            self.assertIn("candidate-1", csv_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
