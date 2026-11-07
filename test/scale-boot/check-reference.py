#!/usr/bin/env python3
"""Validate all frozen references and explicit hand-authored decisions."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from reference import Infeasible, materialize_recipe, search, plan_bytes, reference_plan
from verify import InvalidPlan, read_json, verify

ROOT = Path(__file__).resolve().parent


def load(suite_name, case_name):
    suite = ROOT / "cases" / suite_name
    directory = suite / case_name
    spec_path = suite / "operator-spec.json"
    digest = "sha256:" + hashlib.sha256(spec_path.read_bytes()).hexdigest()
    return (directory, read_json(directory / "graph.json"), read_json(directory / "case.json"),
            read_json(spec_path), digest)


class ReferenceTests(unittest.TestCase):
    def test_all_frozen_references(self):
        with tempfile.TemporaryDirectory(prefix="scale-boot-references-") as temporary:
            for suite, count in (("threshold-q2", 18), ("native-q50", 1), ("strict-small-q50", 8)):
                result = subprocess.run([sys.executable, str(ROOT / "run-references.py"),
                    "--suite", str(ROOT / "cases" / suite), "--output-dir", str(Path(temporary) / suite)],
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                summary = read_json(Path(temporary) / suite / "summary.json")
                self.assertEqual(summary["reference_count"], count)
                self.assertTrue(all(c["manual_valid"] for c in summary["cases"]))

    def test_strict_small_recipes_and_boot_boundaries(self):
        suite = ROOT / "cases/strict-small-q50"
        for name in read_json(suite / "suite.json")["cases"]:
            with self.subTest(case=name):
                directory, graph, case, spec, digest = load("strict-small-q50", name)
                recipe = read_json(directory / "recipe.json")
                plan, info = materialize_recipe(graph, case, spec, digest, recipe)
                self.assertEqual(hashlib.sha256(plan_bytes(plan)).hexdigest(),
                                 read_json(directory / "reference.json")["plan_sha256"])
                self.assertGreaterEqual(read_json(directory / "reference.json")["graph_metrics"]["multiplicative_depth"], 12)
                values = {v["id"]: v for v in plan["values"]}
                for step in plan["execution"]:
                    if step.get("op") == "boot":
                        before, after = values[step["inputs"][0]], values[step["output"]]
                        self.assertEqual((before["level"], after["level"]), (5, 30))
                        self.assertAlmostEqual(before["scale_log2"], 40)
                        self.assertAlmostEqual(after["scale_log2"], 40)
                if info["boot_count"]:
                    recipe["boots_before"] = {}
                    with self.assertRaises((Infeasible, InvalidPlan)):
                        materialize_recipe(graph, case, spec, digest, recipe)

    def test_hand_recipe_reproduces_complete_plan(self):
        directory, graph, case, spec, digest = load("native-q50", "deep-residual")
        recipe = read_json(directory / "recipe.json")
        plan, info = materialize_recipe(graph, case, spec, digest, recipe)
        self.assertEqual(hashlib.sha256(plan_bytes(plan)).hexdigest(),
                         read_json(directory / "reference.json")["plan_sha256"])
        self.assertEqual(info["boot_count"], 1)
        self.assertEqual([e["at"] for e in info["decisions"] if e["op"] == "boot"], ["v9"])
        # Deleting the necessary refresh cannot silently trigger automatic repair.
        recipe["boots_before"] = {}
        with self.assertRaises((Infeasible, InvalidPlan)):
            materialize_recipe(graph, case, spec, digest, recipe)

    def test_small_search_references_remain_feasible(self):
        for name in ("cp-double-rescale", "exhausted-shared-input", "unequal-depth-residual"):
            with self.subTest(name=name):
                directory, graph, case, spec, digest = load("threshold-q2", name)
                plan, info = search(graph, case, spec, digest)
                verify(plan, spec, case, digest)
                self.assertLessEqual(info["boot_count"], read_json(directory / "reference.json")["boot_count"])

    def test_reference_rejects_changed_computation(self):
        directory, _, case, spec, digest = load("threshold-q2", "cp-double-rescale")
        plan = copy.deepcopy(reference_plan(directory))
        encode = next(s for s in plan["execution"] if s["kind"] == "encode")
        encode["payload"]["values"] = [1.02] * 8
        with self.assertRaisesRegex(InvalidPlan, "test vector"):
            verify(plan, spec, case, digest)

    def test_regenerated_plan_must_match_frozen_hash(self):
        source = ROOT / "cases/threshold-q2/shared-square"
        with tempfile.TemporaryDirectory() as temporary:
            suite = Path(temporary)
            directory = suite / source.name
            directory.mkdir()
            (suite / "operator-spec.json").write_bytes((source.parent / "operator-spec.json").read_bytes())
            for name in ("graph", "case", "reference"):
                (directory / f"{name}.json").write_bytes((source / f"{name}.json").read_bytes())
            reference_plan(directory)
            metadata = read_json(directory / "reference.json")
            metadata["plan_sha256"] = "0" * 64
            (directory / "reference.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "regenerated reference plan hash mismatch"):
                reference_plan(directory)

    def test_small_generator_emits_compact_references(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run([sys.executable, str(ROOT / "build-small.py"),
                "--output-dir", temporary], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            suite = Path(temporary)
            self.assertFalse(list(suite.glob("*/manual.json")))
            for name in read_json(suite / "suite.json")["cases"]:
                reference_plan(suite / name)


if __name__ == "__main__":
    unittest.main()
