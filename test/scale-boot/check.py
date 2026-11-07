#!/usr/bin/env python3
"""Adversarial tests for the independent scale/Boot verifier."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from example import make_example
from verify import InvalidPlan, compare, read_json, verify


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.case, self.spec, self.spec_text, self.digest, self.manual, self.automatic = make_example()

    def check(self, plan):
        return verify(plan, self.spec, self.case, self.digest)

    def test_valid_comparison_and_counts(self):
        report = compare(self.manual, self.automatic, self.spec, self.case, self.digest)
        self.assertEqual(report["boot_delta"], 1)
        self.assertEqual(report["comparison"], "worse")
        stats = report["plans"]["manual"]["stats"]
        self.assertEqual(stats["rescale_levels"], 1)
        self.assertEqual(stats["test_vector_count"], 2)
        self.assertEqual(stats["plaintext_max_abs_error"], 0)
        better = compare(self.automatic, self.manual, self.spec, self.case, self.digest)
        self.assertEqual(better["comparison"], "better")

    def test_reject_state_and_structure_errors(self):
        def change_value(field, value):
            return lambda p: p["values"][3].__setitem__(field, value)
        mutations = [
            ("duplicate", lambda p: p["values"].append(deepcopy(p["values"][0]))),
            ("before definition", lambda p: p["execution"][0].__setitem__("inputs", ["0", "4"])),
            ("redefined", lambda p: p["execution"][0].__setitem__("output", "0")),
            ("ordinals", lambda p: p["execution"][0].__setitem__("ordinal", 9)),
            ("capacity", change_value("scale_log2", 100)),
            ("floor", change_value("scale_log2", 1)),
            ("outside bounds", change_value("level", -1)),
            ("output state", change_value("scale_log2", 9.5)),
            ("NTT state", change_value("ntt", False)),
            ("actual moduli", lambda p: p["execution"][2]["attrs"].__setitem__("target_scale_log2", 9.5)),
            ("must decrease", lambda p: p["execution"][2]["attrs"].__setitem__("target_level", 2)),
            ("input level", lambda p: p["values"][0].__setitem__("level", 1)),
            ("output state", lambda p: p["values"][1].__setitem__("components", 2)),
            ("unsupported step", lambda p: p["execution"][0].__setitem__("kind", "copy")),
            ("identity/hash", lambda p: p["target"]["operator_spec"].__setitem__("source_sha256", "bad")),
        ]
        for message, mutate in mutations:
            with self.subTest(message=message):
                plan = deepcopy(self.manual)
                mutate(plan)
                with self.assertRaisesRegex(InvalidPlan, message):
                    self.check(plan)

    def test_reject_wrong_rotation_even_with_valid_states(self):
        self.manual["execution"][3]["attrs"]["steps"] = -1
        with self.assertRaisesRegex(InvalidPlan, "test vector"):
            self.check(self.manual)

    def test_reject_wrong_output_contract(self):
        self.case["outputs"][0]["min_level"] = 2
        with self.assertRaisesRegex(InvalidPlan, "output level/scale"):
            self.check(self.manual)

    def test_boot_contracts(self):
        for key, value, message in (("input_level_min", 2, "Boot input contract"),
                                     ("input_scale_log2", 10, "Boot input scale"),
                                     ("output_scale_log2", 11, "Boot output contract")):
            with self.subTest(key=key):
                spec = deepcopy(self.spec)
                spec["boot_profiles"][0][key] = value
                with self.assertRaisesRegex(InvalidPlan, message):
                    verify(self.automatic, spec, self.case, self.digest)

    def test_invalid_plan_is_never_a_better_result(self):
        self.manual["execution"][3]["attrs"]["steps"] = -1
        report = compare(self.automatic, self.manual, self.spec, self.case, self.digest)
        self.assertEqual(report["comparison"], "invalid")
        self.assertIsNone(report["boot_delta"])

    def test_inline_units_precision_and_zero_padding(self):
        plan = deepcopy(self.manual)
        plain = dict(plan["values"][-1], id="6", kind="plaintext", components=1, scale_log2=3)
        result = dict(plan["values"][-1], id="7", scale_log2=plan["values"][-1]["scale_log2"] + 3)
        plan["values"] += [plain, result]
        plan["initialization"] = [{"ordinal": 0, "kind": "encode", "output": "6",
                                    "payload": {"kind": "inline", "values": [1, 1, 1, 1]}}]
        for step in plan["execution"]:
            step["ordinal"] += 1
        plan["execution"].append({"ordinal": 5, "kind": "compute", "op": "mul_cp",
            "inputs": ["4", "6"], "output": "7", "place": dict(plain["place"])})
        plan["final_outputs"] = ["7"]
        self.case["outputs"][0]["max_scale_log2"] = 16
        self.assertEqual(self.check(plan)["encode_count"], 1)
        self.case["min_unit_plaintext_scale_log2"] = 4
        with self.assertRaisesRegex(InvalidPlan, "plaintext encoding scale"):
            self.check(plan)
        self.case["min_plaintext_scale_log2"] = 3
        plan["initialization"][0]["payload"]["values"] = [1]
        with self.assertRaisesRegex(InvalidPlan, "test vector"):
            self.check(plan)

    def test_disabled_operator_and_nonfinite_replay(self):
        self.spec["operators"]["rotate"]["supported"] = False
        with self.assertRaisesRegex(InvalidPlan, "operator disabled"):
            self.check(self.manual)
        self.spec["operators"]["rotate"]["supported"] = True
        self.case["test_vectors"][0]["inputs"][0][0] = 1e308
        with self.assertRaisesRegex(InvalidPlan, "non-finite plaintext replay"):
            self.check(self.manual)

    def test_nonfinite_and_duplicate_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for text in ('{"x": 1, "x": 2}', '{"x": NaN}'):
                path.write_text(text)
                with self.assertRaises(InvalidPlan):
                    read_json(path)

    def test_cli_reports_and_exit_codes(self):
        root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            subprocess.run([sys.executable, str(root / "example.py"), "--output-dir", str(out)], check=True)
            command = [sys.executable, str(root / "verify.py"), "--case", str(out / "case.json"),
                "--operator-spec", str(out / "operator-spec.json"), "--manual", str(out / "manual.json"),
                "--automatic", str(out / "automatic.json"), "--output", str(out / "report.json")]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads(run.stdout)
            self.assertEqual(report["boot_delta"], 1)
            self.assertEqual(report, read_json(out / "report.json"))
            run = subprocess.run(command + ["--fail-on-worse"], capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            (out / "automatic.json").write_text('{}')
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 1)
            self.assertIsNone(json.loads(run.stdout)["boot_delta"])


if __name__ == "__main__":
    unittest.main()
