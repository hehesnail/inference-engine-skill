import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, Mock, patch


HELPER = (
    Path(__file__).resolve().parents[1] / "skills" / "inference-engine"
    / "library" / "ncu-report" / "helpers" / "analyze_reports.py"
)


class NcuReportCompatibilityTest(unittest.TestCase):
    def test_metrics_survive_a_binding_without_rule_results(self):
        with patch.dict(sys.modules, {"ncu_report": ModuleType("ncu_report")}), patch.object(sys, "path", sys.path.copy()):
            collect = runpy.run_path(str(HELPER))["collect"]
        interfaces = ["name", "NameBase_FUNCTION", "NameBase_DEMANGLED", "NameBase_MANGLED", "metric_names", "__getitem__"]
        metric_name = "dram__bytes_read.sum"
        metric = Mock()
        metric.value.return_value = 256
        metric.unit.return_value = "byte"
        for exposes_rules in (False, True):
            with self.subTest(exposes_rules=exposes_rules), tempfile.TemporaryDirectory() as temp:
                action = MagicMock(spec=interfaces + (["rule_results"] if exposes_rules else []))
                action.name.return_value = "example_kernel"
                action.metric_names.return_value = [metric_name]
                action.__getitem__.side_effect = {metric_name: metric}.__getitem__
                if exposes_rules:
                    action.rule_results.return_value = []
                folder = Path(temp)
                collect(action, "test.ncu-rep", "sample", folder)
                archive = json.loads((folder / "metrics_all_sample.json").read_text())
                self.assertEqual(archive, [{"name": metric_name, "value": 256, "unit": "byte"}])
                rules = (folder / "rules_sample.txt").read_text()
                self.assertEqual("unavailable" in rules, not exposes_rules)
                if exposes_rules:
                    action.rule_results.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
