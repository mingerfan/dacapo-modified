#!/usr/bin/env python3
"""Check graph completeness, matching, failure display, and stale-data guards."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from draw_graphs import SVG, original_model, plan_model, render_suite
from verify import read_json
from reference import plan_bytes, reference_plan

ROOT = Path(__file__).resolve().parent
SUITE = ROOT / "cases/threshold-q2"


class GraphTests(unittest.TestCase):
    def test_every_operation_and_dependency_is_preserved(self):
        directories = [suite / name for suite in (SUITE, ROOT / "cases/strict-small-q50")
                       for name in read_json(suite / "suite.json")["cases"]]
        for directory in directories:
            with self.subTest(case=str(directory)):
                spec = read_json(directory.parent / "operator-spec.json")
                graph, case = read_json(directory / "graph.json"), read_json(directory / "case.json")
                original, origins = original_model(graph, case, spec)
                self.assertEqual(len(original["nodes"]), len(graph["inputs"]) + len(graph["constants"]) +
                                 len(graph["ops"]) + len(graph["outputs"]))
                plan = reference_plan(directory)
                model = plan_model(plan, case, spec, origins)
                steps = plan["initialization"] + plan["execution"] + plan["finalization"]
                self.assertEqual(len(model["nodes"]), len(plan["external_inputs"]) + len(steps) + len(plan["final_outputs"]))
                self.assertEqual(len(model["edges"]), sum(len(set(s.get("inputs", []))) for s in steps) + len(plan["final_outputs"]))
                self.assertEqual(model["boot_count"], sum(s.get("op") == "boot" for s in steps))
                self.assertTrue(all(n["origins"] for n in model["nodes"] if n["kind"] == "compute"))

    def test_does_not_guess_correspondence(self):
        directory = SUITE / "cp-double-rescale"
        spec, case = read_json(SUITE / "operator-spec.json"), read_json(directory / "case.json")
        _, origins = original_model(read_json(directory / "graph.json"), case, spec)
        plan = copy.deepcopy(reference_plan(directory))
        step = next(s for s in plan["execution"] if s["kind"] == "encode")
        step["payload"]["values"] = [1.02] * 8
        model = plan_model(plan, case, spec, origins)
        self.assertTrue(any(not n["origins"] for n in model["nodes"] if n["kind"] == "compute"))

    def test_render_statuses_and_checksums(self):
        with tempfile.TemporaryDirectory(prefix="scale-boot-graphs-") as temporary:
            root = Path(temporary)
            suite, results, output = root / "suite", root / "results", root / "graphs"
            name = "shared-square"
            directory = suite / name
            directory.mkdir(parents=True)
            result_dir = results / name
            result_dir.mkdir(parents=True)
            for item in ("graph", "case", "reference"):
                (directory / f"{item}.json").write_bytes((SUITE / name / f"{item}.json").read_bytes())
            (suite / "operator-spec.json").write_bytes((SUITE / "operator-spec.json").read_bytes())
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            row = dict(case=name, reference_boots=1, input_compute_ops=3,
                artifacts_sha256={k: digest(directory / f"{k}.json") for k in ("graph", "case")}, algorithms={})
            manual_bytes = plan_bytes(reference_plan(directory))
            row["artifacts_sha256"]["manual"] = hashlib.sha256(manual_bytes).hexdigest()
            for strategy, status in (("greedy", "valid"), ("depth-dp", "invalid")):
                plan = result_dir / f"{strategy}.graph.runtime-plan.json"
                plan.write_bytes(manual_bytes)
                row["algorithms"][strategy] = dict(status=status, plan_sha256=digest(plan), stats={"boot_count": 1})
                if status == "invalid":
                    row["algorithms"][strategy]["error"] = "precision floor rejected"
            row["algorithms"]["optimized"] = dict(status="compile_failed", error="no feasible prefix")
            # Failed compilation must not display a stale plan left by an older run.
            (result_dir / "optimized.graph.runtime-plan.json").write_text("stale invalid JSON")
            summary = dict(suite="fixture", operator_spec_sha256="sha256:" + digest(suite / "operator-spec.json"), cases=[row])
            (results / "summary.json").write_text(json.dumps(summary))
            render_suite(suite, results, output)
            page = (output / name / "index.html").read_text()
            self.assertIn("编译失败", page)
            self.assertIn("未通过校验", page)
            self.assertIn("precision floor rejected", page)
            self.assertIn("s=UNRESTRICTED", page)
            self.assertIn("L+1", page)
            self.assertNotIn("stale invalid JSON", page)
            root_svg = ET.parse(output / name / "manual.svg").getroot()
            nodes = [e for e in root_svg.iter(f"{{{SVG}}}g") if e.get("data-kind")]
            metadata = read_json(output / name / "graph-metadata.json")
            self.assertEqual(len(nodes), metadata["manual"]["node_count"])
            self.assertEqual(sum(n.get("data-kind") == "boot" for n in nodes), 1)
            self.assertTrue((output / name / "comparison.svg").is_file())
            combined = ET.parse(output / name / "comparison.svg").getroot()
            # Original + reference + two diagnostic/valid plans each square twice.
            self.assertEqual(sum(e.text == "x2" for e in combined.iter(f"{{{SVG}}}text")), 8)
            self.assertEqual(metadata["optimized"]["node_count"], 0)
            plan_path = result_dir / "greedy.graph.runtime-plan.json"
            plan_path.write_text(plan_path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "automatic plan changed"):
                render_suite(suite, results, output)
            plan_path.write_bytes(manual_bytes)
            (directory / "graph.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "graph changed"):
                render_suite(suite, results, output)


if __name__ == "__main__":
    unittest.main()
