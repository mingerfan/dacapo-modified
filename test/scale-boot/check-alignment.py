#!/usr/bin/env python3
"""Regression tests for unit quantization and fixed-boundary alignment budgets."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from alignment import verify_alignment_budget
from alignment_example import SUITE, make_fixtures, run
from example import make_example
from verify import InvalidPlan, decoded_unit, read_json, verify


class AlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = read_json(SUITE / "operator-spec.json")
        cls.digest = "sha256:" + hashlib.sha256((SUITE / "operator-spec.json").read_bytes()).hexdigest()

    def load(self, name="reuse-modswitch"):
        return {stem: read_json(SUITE / name / f"{stem}.json")
                for stem in ("case", "before", "candidate", "contract", "expected")}

    def check(self, fixture, variant="candidate"):
        return verify(fixture[variant], self.spec, fixture["case"], self.digest)

    def test_three_frozen_recipes_and_expected_outcomes(self):
        text, generated = make_fixtures()
        self.assertEqual(text, (SUITE / "operator-spec.json").read_text())
        for name, fixture in generated.items():
            self.assertEqual(fixture, self.load(name), name)
        with tempfile.TemporaryDirectory() as temp:
            report = run(SUITE, Path(temp))
        self.assertEqual([r["accepted"] for r in report["cases"]], [True, True, False])
        absorbed, reused, refused = report["cases"]
        self.assertEqual(absorbed["budget"]["stats"]["operation_delta"]["mul_cp"], -1)
        self.assertEqual(absorbed["budget"]["stats"]["operation_delta"]["rescale"], 0)
        self.assertEqual(reused["budget"]["stats"]["operation_delta"]["rescale"], 1)
        self.assertEqual(reused["budget"]["stats"]["operation_delta"]["mod_switch"], -1)
        for row in (absorbed, reused):
            self.assertTrue(all(b["level_delta"] == 0 for b in row["budget"]["stats"]["boundaries"]))
            self.assertEqual(row["candidate"]["stats"]["boot_count"], 0)
        self.assertTrue(refused["candidate"]["valid"])
        self.assertIn("level changed 5 -> 4", refused["budget"]["error"])

    def test_unit_rounding_matches_positive_cpp_half_ties(self):
        for s in (0, 1, 11, 30, 53):
            self.assertEqual(decoded_unit(s), 1.)
        self.assertAlmostEqual(decoded_unit(math.log2(1.5)), 2/1.5)
        self.assertAlmostEqual(decoded_unit(.2), 1/math.exp2(.2))
        for invalid in (-1, float("nan"), float("inf"), 1024):
            with self.assertRaises(InvalidPlan):
                decoded_unit(invalid)

    def test_bad_unit_was_invisible_to_ideal_replay(self):
        fixture = self.load()
        with self.assertRaisesRegex(InvalidPlan, "unit encoding error"):
            self.check(fixture, "before")
        legacy = deepcopy(fixture)
        del legacy["case"]["unit_encoding"]
        legacy["case"]["min_unit_plaintext_scale_log2"] = 0
        stats = self.check(legacy, "before")
        self.assertEqual(stats["plaintext_max_abs_error"], 0)
        self.assertFalse(stats["unit_quantization_checked"])
        # Even if the local budget permits it, the propagated error must fail.
        fixture["case"]["unit_encoding"]["max_abs_error"] = 1
        with self.assertRaisesRegex(InvalidPlan, "unit-quantized test vector"):
            self.check(fixture, "before")

    def test_unit_error_propagation_and_cancellation(self):
        fixture = self.load()
        stats = self.check(fixture)
        self.assertEqual(stats["plaintext_max_abs_error"], 0)
        # Second input vector includes exact cancellation; relative-only checks
        # would be unsuitable. Independently compute the scalar rounding error.
        q = self.spec["context"]["rns_moduli"][6]
        encoded_scale = math.exp2(40.2 + math.log2(q) - 40)
        factor = math.floor(encoded_scale + .5) / encoded_scale
        expected = max(abs(a*factor+b-(a+b)) for vector in fixture["case"]["test_vectors"]
                       for a, b in zip(*vector["inputs"]))
        self.assertGreater(expected, 0)
        self.assertAlmostEqual(stats["unit_quantized_max_abs_error"], expected, places=15)
        fixture["case"]["unit_encoding"]["output_tolerance"] = dict(abs=1e-12, rel=1e-12)
        with self.assertRaisesRegex(InvalidPlan, "unit-quantized test vector"):
            self.check(fixture)

    def test_exact_small_unit_is_not_rejected_by_legacy_floor(self):
        fixture = self.load()
        plan = fixture["before"]
        for v in plan["values"]:
            if v["kind"] == "plaintext":
                v["scale_log2"] = 0
            elif abs(v["scale_log2"] - 40.2) < 1e-8:
                v["scale_log2"] = 40
        fixture["case"]["inputs"][1]["scale_log2"] = 40
        stats = self.check(fixture, "before")
        self.assertEqual(stats["unit_encodings"][0]["abs_error"], 0)
        self.assertEqual(stats["unit_quantized_max_abs_error"], 0)

    def test_minus_one_and_mask_are_distinguished(self):
        fixture = self.load()
        encode = next(s for s in fixture["candidate"]["execution"] if s["kind"] == "encode")
        encode["payload"]["values"] = [-1.] * 8
        for vector in fixture["case"]["test_vectors"]:
            vector["outputs"] = [[-a+b for a, b in zip(*vector["inputs"])]]
        self.assertTrue(self.check(fixture)["unit_quantization_checked"])
        # A short [1] is zero padded; it is a mask, not full-slot +1.
        encode["payload"]["values"] = [1.]
        with self.assertRaisesRegex(InvalidPlan, "plaintext encoding scale"):
            self.check(fixture)

    def test_budget_checks_all_boundaries_not_just_outputs(self):
        fixture = self.load()
        plan = fixture["candidate"]
        contract = dict(format_version=1, boundaries=[dict(name=v["id"], before=v["id"], after=v["id"])
            for v in plan["values"] if v["kind"] == "ciphertext"])
        changed = deepcopy(plan)
        intermediate = changed["execution"][1]["output"]
        next(v for v in changed["values"] if v["id"] == intermediate)["level"] -= 1
        with self.assertRaisesRegex(InvalidPlan, "level changed"):
            verify_alignment_budget(plan, changed, contract)
        for field, value in (("scale_log2", 40.3), ("components", 3), ("ntt", False)):
            changed = deepcopy(plan)
            changed["values"][-1][field] = value
            with self.assertRaisesRegex(InvalidPlan, field + " changed"):
                verify_alignment_budget(plan, changed, contract)

    def test_missing_or_duplicate_boundary_binding_is_rejected(self):
        fixture = self.load()
        for bindings, message in ((fixture["contract"]["boundaries"][:-1], "unbound final_outputs"),
                                  (fixture["contract"]["boundaries"] * 2, "duplicate/invalid")):
            with self.assertRaisesRegex(InvalidPlan, message):
                verify_alignment_budget(fixture["before"], fixture["candidate"],
                                        dict(format_version=1, boundaries=bindings))

    def test_boot_contract_and_bound_endpoints(self):
        _, _, _, _, _, plan = make_example()
        bindings = [dict(name=v["id"], before=v["id"], after=v["id"]) for v in plan["values"]]
        contract = dict(format_version=1, boundaries=bindings)
        self.assertEqual(verify_alignment_budget(plan, plan, contract)["boot_count"], 1)
        changed = deepcopy(plan)
        changed["execution"][-1]["attrs"]["operator_profile"] = "another-profile"
        with self.assertRaisesRegex(InvalidPlan, "Boot contract changed"):
            verify_alignment_budget(plan, changed, contract)
        changed = deepcopy(plan)
        changed["execution"][-1]["inputs"] = ["3"]
        with self.assertRaisesRegex(InvalidPlan, "Boot binding changed"):
            verify_alignment_budget(plan, changed, contract)
        changed = deepcopy(plan)
        changed["execution"].pop()
        with self.assertRaisesRegex(InvalidPlan, "Boot count changed"):
            verify_alignment_budget(plan, changed, contract)
        contract["boundaries"] = [b for b in bindings if b["before"] != "4"]
        with self.assertRaisesRegex(InvalidPlan, "unbound Boot endpoint"):
            verify_alignment_budget(plan, plan, contract)

    def test_invalid_error_policy_fails_explicitly(self):
        for policy in (dict(max_abs_error=-1, output_tolerance=dict(abs=0, rel=0)),
                       dict(max_abs_error=1e-6, output_tolerance=dict(abs=float("nan"), rel=0)),
                       dict(max_abs_error=1e-6)):
            fixture = self.load()
            fixture["case"]["unit_encoding"] = policy
            with self.assertRaises(InvalidPlan):
                self.check(fixture)

    def test_cli_example_runs_without_graphviz(self):
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("alignment_example.py")),
                "--output-dir", temp], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(all(row["expectation_met"] for row in read_json(Path(temp)/"summary.json")["cases"]))


if __name__ == "__main__":
    unittest.main()
