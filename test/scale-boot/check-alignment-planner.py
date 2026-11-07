#!/usr/bin/env python3
"""Exercise the C++ core and independently replay its materialized plans."""
import argparse
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import tempfile
import unittest

from alignment import steps, verify_alignment_budget
from alignment_example import SUITE
from alignment_planner import extract, query, run_fixtures, solve
from reference import Plan
from verify import read_json, verify

DRIVER = None


class PlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = read_json(SUITE / "operator-spec.json")
        cls.digest = "sha256:" + hashlib.sha256((SUITE / "operator-spec.json").read_bytes()).hexdigest()

    def fixture(self, name="absorb-mulcp"):
        before = read_json(SUITE / name / "before.json")
        case = read_json(SUITE / name / "case.json")
        root = next(s for s in reversed(steps(before)) if s.get("op") == "mul_cp")["output"]
        return before, case, root

    def request(self, name="absorb-mulcp"):
        before, case, root = self.fixture(name)
        return extract(before, root, self.spec, case)[0]

    def test_three_frozen_cases_use_automatic_cpp_plans(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_fixtures(DRIVER, Path(directory))
        absorbed, explicit, failed = report["cases"]
        self.assertEqual(absorbed["search"]["plan"]["strategy"], "absorb-mulcp")
        self.assertEqual(explicit["search"]["plan"]["strategy"], "unit-rescale")
        self.assertEqual(failed["search"]["status"], "no-candidate")
        self.assertNotIn("plan", failed["search"])
        for row in (absorbed, explicit):
            self.assertEqual(row["verification"]["boot_count"], 0)
            self.assertTrue(all(b["level_delta"] == 0 for b in row["budget"]["boundaries"]))
        self.assertEqual(absorbed["budget"]["operation_delta"]["mul_cp"], -1)
        self.assertEqual(explicit["budget"]["operation_delta"]["rescale"], 1)

    def test_search_and_materialization_leave_source_unchanged(self):
        for name in ("absorb-mulcp", "reuse-modswitch", "insufficient-budget"):
            before, case, root = self.fixture(name)
            snapshot = deepcopy(before)
            result, candidate, _, request = solve(before, root, self.spec, case, DRIVER)
            self.assertEqual(before, snapshot)
            self.assertEqual(query(DRIVER, request), result)  # deterministic
            self.assertIsNot(candidate, before)

    def extended(self):
        _, case, _ = self.fixture()
        p = Plan(self.spec, case, self.digest)
        x, y = p.external
        removed = sum(math.log2(self.spec["context"]["rns_moduli"][i]) for i in (8, 7))
        w = p.encode(.75, 8, removed)
        x = p.compute("mul_cp", [x, w], 8, 40+removed)
        for level in (8, 7):
            scale = p.state(x)["scale_log2"] - math.log2(self.spec["context"]["rns_moduli"][level])
            x = p.compute("rescale", [x], level-1, scale, target_level=level-1, target_scale_log2=scale)
        scale = p.state(x)["scale_log2"]
        x = p.compute("rotate", [x], 6, scale, steps=2)
        bias = p.encode(.3, 6, scale)
        x = p.compute("add_cp", [x, bias], 6, scale)
        x = p.compute("negate", [x], 6, scale)
        unit = p.encode(1., 6, 40.2-scale)
        root = p.compute("mul_cp", [x, unit], 6, 40.2)
        p.bindings["out"] = p.compute("add_cc", [root, y], 6, 40.2)
        for v in case["test_vectors"]:
            a, b = v["inputs"]
            rotated = a[2:] + a[:2]
            v["outputs"] = [[-(.75*t+.3)+u for t, u in zip(rotated, b)]]
        return p.export(["out"]), case, root

    def test_reverse_across_rescale_rotate_addcp_negate(self):
        before, case, root = self.extended()
        result, automatic, _, _ = solve(before, root, self.spec, case, DRIVER)
        self.assertEqual(result["plan"]["strategy"], "absorb-mulcp")
        self.assertEqual(result["plan"]["reencoded_plaintexts"], 2)
        stats = verify(automatic, self.spec, case, self.digest)
        self.assertLessEqual(stats["plaintext_max_abs_error"], 1e-12)
        payloads = [s["payload"]["values"] for s in steps(automatic) if s["kind"] == "encode"]
        self.assertEqual(payloads, [[.75]*8, [.3]*8])
        self.assertEqual(next(s for s in steps(automatic) if s.get("op") == "rotate")["attrs"], {"steps": 2})
        contract = dict(format_version=1, boundaries=[dict(name=v, before=v, after=v)
            for v in before["external_inputs"]+before["final_outputs"]])
        verify_alignment_budget(before, automatic, contract)

    def test_shared_or_protected_ciphertext_stops_backtracking(self):
        before, case, root = self.fixture()
        source = next(s for s in steps(before) if s["output"] == root)["inputs"][0]
        result, automatic, _, request = solve(before, root, self.spec, case, DRIVER, protected=[source])
        self.assertEqual(request["operations"], [])
        self.assertEqual(result["status"], "no-candidate")
        self.assertIsNone(automatic)
        before["final_outputs"].append(source)
        result, _, _, request = solve(before, root, self.spec, case, DRIVER)
        self.assertEqual(request["operations"], [])
        self.assertEqual(result["status"], "no-candidate")

    def test_new_unit_is_inserted_before_rotation_and_bias(self):
        before, case, root = self.extended()
        # Turn the first two RS into one ModSwitch, replacing the weight
        # multiplication with an input boundary at s40. The suffix must remain.
        request, _ = extract(before, root, self.spec, case)
        request["operations"] = [dict(op="mod_switch", output=dict(level=6, scale_log2=40))] + request["operations"][3:]
        result = query(DRIVER, request)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["plan"]["strategy"], "unit-rescale")
        rows = result["plan"]["steps"]
        add = next(s for s in rows if s["op"] == "add_cp")
        self.assertAlmostEqual(add["plaintext_scale"], 40.2)
        self.assertLess(next(i for i,s in enumerate(rows) if s["op"] == "rescale"),
                        next(i for i,s in enumerate(rows) if s["op"] == "rotate"))

    def test_small_exact_unit_requires_no_rescale(self):
        request = self.request("insufficient-budget")
        request["target"]["scale_log2"] = 41
        result = query(DRIVER, request)
        self.assertEqual(result["plan"]["strategy"], "explicit-unit")
        self.assertEqual(result["plan"]["added_rescale_levels"], 0)
        self.assertEqual(result["plan"]["steps"][0]["plaintext_scale"], 1)

    def test_insert_unit_before_existing_rescale_without_new_level_cost(self):
        before, case, root = self.fixture("reuse-modswitch")
        desc = {v["id"]: v for v in before["values"]}
        rs = next(s for s in steps(before) if s.get("op") == "mod_switch")
        raw_scale = 40 - math.log2(self.spec["context"]["rns_moduli"][6])
        rs["op"] = "rescale"
        rs["attrs"]["target_scale_log2"] = raw_scale
        desc[rs["output"]]["scale_log2"] = raw_scale
        unit = next(s for s in steps(before) if s["output"] == root)
        desc[unit["inputs"][1]]["scale_log2"] = 40.2-raw_scale
        # The original scheduled RS is structurally meaningful but cannot
        # execute at the configured floor until the unit is hoisted before it.
        snapshot = deepcopy(before)
        result, automatic, _, _ = solve(before, root, self.spec, case, DRIVER)
        self.assertEqual(before, snapshot)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["plan"]["strategy"], "explicit-unit")
        self.assertEqual(result["plan"]["site"], 0)
        self.assertEqual(result["plan"]["added_rescale_levels"], 0)
        self.assertEqual([s["op"] for s in result["plan"]["steps"]], ["mul_cp", "rescale"])
        self.assertEqual(result["plan"]["steps"][1]["origin"], 0)
        verify(automatic, self.spec, case, self.digest)
        contract = dict(format_version=1, boundaries=[dict(name=v, before=v, after=v)
            for v in before["external_inputs"]+before["final_outputs"]])
        budget = verify_alignment_budget(before, automatic, contract)
        self.assertEqual(budget["operation_delta"]["rescale"], 0)
        self.assertEqual(budget["operation_delta"]["boot"], 0)

    def test_existing_rescale_does_not_automatically_improve_small_correction(self):
        request = self.request("reuse-modswitch")
        q = math.log2(request["moduli"][6])
        request["input"]["scale_log2"] = 40+q
        request["operations"] = [dict(op="rescale", output=dict(level=5, scale_log2=40))]
        # Both before-RS and after-RS insertion still require s=0.2. Hoisting
        # alone must not be mistaken for a high-precision unit encoding.
        result = query(DRIVER, request)
        self.assertEqual(result["status"], "no-candidate")
        self.assertEqual(result["rejected"]["plaintext_precision"], 2)

    def test_already_aligned_does_not_add_work(self):
        for name in ("reuse-modswitch", "absorb-mulcp"):
            request = self.request(name)
            request["target"]["scale_log2"] = 40
            result = query(DRIVER, request)
            self.assertEqual(result["plan"]["strategy"], "unchanged")
            self.assertEqual(result["plan"]["added_multiplications"], 0)
            self.assertEqual(result["candidates"], 1)

    def test_real_moduli_not_nominal_bit_widths(self):
        request = self.request("reuse-modswitch")
        result = query(DRIVER, request)
        plain = next(s["plaintext_scale"] for s in result["plan"]["steps"] if s["op"] == "mul_cp")
        expected = 40.2 + math.log2(request["moduli"][6]) - 40
        self.assertAlmostEqual(plain, expected)
        self.assertGreater(abs(plain - 32.2), 1e-4)

    def test_searches_higher_prime_window(self):
        request = self.request("reuse-modswitch")
        # Synthetic arithmetic chain isolates prime-window selection, not a
        # deployable CKKS parameter set. The low windows lack precision.
        request.update(moduli=[2**30, 2**30, 4, 8, 2**32],
                       input=dict(level=4, scale_log2=40), target=dict(level=1, scale_log2=40.2),
                       operations=[dict(op="mod_switch", output=dict(level=1, scale_log2=40))])
        request["policy"]["minimum_level"] = 0
        result = query(DRIVER, request)
        self.assertEqual(result["plan"]["added_rescale_levels"], 1)
        rs = next(s for s in result["plan"]["steps"] if s["op"] == "rescale")
        self.assertEqual(rs["output"]["level"], 3)  # q4, not q2
        self.assertEqual(result["plan"]["steps"][-1]["output"]["level"], 1)

    def test_two_rescales_when_one_is_not_precise_enough(self):
        request = self.request("reuse-modswitch")
        request.update(moduli=[2**30]*6, input=dict(level=5, scale_log2=40),
                       target=dict(level=3, scale_log2=40.2),
                       operations=[dict(op="mod_switch", output=dict(level=3, scale_log2=40))])
        request["policy"].update(minimum_level=0, maximum_unit_error=1e-12)
        result = query(DRIVER, request)
        self.assertEqual(result["plan"]["added_rescale_levels"], 2)
        self.assertEqual(result["plan"]["output"]["level"], 3)

    def test_ordinary_weight_cannot_use_unit_precision_rule(self):
        request = self.request("reuse-modswitch")
        request["operations"] = [dict(op="mul_cp", output=dict(level=6, scale_log2=80),
                                      plaintext_scale=40, unit_plaintext=False)]
        request["target"] = dict(level=6, scale_log2=50)
        result = query(DRIVER, request)
        self.assertEqual(result["status"], "no-candidate")
        self.assertGreater(result["rejected"]["plaintext_precision"], 0)
        request["operations"][0]["unit_plaintext"] = True
        result = query(DRIVER, request)
        self.assertEqual(result["plan"]["steps"][0]["plaintext_scale"], 10)

    def test_invalid_capacity_and_exit_level_are_explicit(self):
        for field, value in (("level", 4), ("scale_log2", 1000)):
            request = self.request("reuse-modswitch")
            request["target"][field] = value
            self.assertEqual(query(DRIVER, request)["status"], "invalid-input")
        request = self.request()
        request["operations"][1]["output"]["scale_log2"] += .1
        self.assertIn("actual-prime replay", query(DRIVER, request)["diagnostic"])

    def test_search_limits_are_not_reported_as_infeasible(self):
        for key in ("maximum_candidates", "maximum_chain_length"):
            request = self.request()
            request["policy"][key] = 1
            result = query(DRIVER, request)
            self.assertEqual(result["status"], "search-limit")
            self.assertNotIn("plan", result)

    def test_explicit_conversion_matches_small_exhaustive_oracle(self):
        # Independently enumerate no-RS and all actual-prime RS windows. The
        # oracle checks each intermediate state, not the planner's cost fields.
        for source in range(6, 9):
            for target in range(4, source + 1):
                for incoming, wanted in ((40., 40.2), (40., 41.), (60., 40.3)):
                    request = self.request("reuse-modswitch")
                    request.update(input=dict(level=source, scale_log2=incoming),
                                   target=dict(level=target, scale_log2=wanted),
                                   operations=[] if target == source else [dict(op="mod_switch",
                                       output=dict(level=target, scale_log2=incoming))])
                    request["policy"]["maximum_unit_error"] = 1e-10
                    moduli = request["moduli"]
                    capacity = lambda level: sum(math.log2(q) for q in moduli[:level+1])
                    feasible = []
                    for count in range(source-target+1):
                        for high in range(target+count, source+1):
                            plain = wanted-incoming + sum(math.log2(moduli[q])
                                                          for q in range(high-count+1, high+1))
                            if not 0 <= plain <= 1023:
                                continue
                            scale = 2**plain
                            if abs(math.floor(scale+.5)/scale-1) > 1e-10:
                                continue
                            current = incoming+plain
                            if current+1 >= capacity(high):
                                continue
                            valid = True
                            for at in range(high, high-count, -1):
                                current -= math.log2(moduli[at])
                                valid &= current >= 40-1e-8 and current+1 < capacity(at-1)
                            if valid and abs(current-wanted) < 1e-8 and current+1 < capacity(target):
                                feasible.append(count)
                    result = query(DRIVER, request)
                    with self.subTest(source=source, target=target, incoming=incoming, wanted=wanted):
                        self.assertEqual(result["status"], "success" if feasible else "no-candidate")
                        if feasible:
                            self.assertEqual(result["plan"]["added_rescale_levels"], min(feasible))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    DRIVER = args.driver
    unittest.main(argv=[__file__] + remaining)
