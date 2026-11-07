#!/usr/bin/env python3
"""Validate source-derived fixtures, vector lowering, and shared Boot contracts."""
import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from reference import materialize_recipe, plan_bytes, reference_plan
from verify import InvalidPlan, read_json, verify

ROOT = Path(__file__).resolve().parent
SUITE = ROOT/'cases/model-large-q50'


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


large = module('large_patterns', 'build-large.py')
runner = module('reference_runner', 'run-references.py')


class LargeTests(unittest.TestCase):
    def test_all_frozen_recipes_and_boot_contracts(self):
        spec = read_json(SUITE/'operator-spec.json')
        digest = 'sha256:'+hashlib.sha256((SUITE/'operator-spec.json').read_bytes()).hexdigest()
        self.assertEqual(spec['boot_profiles'], read_json(ROOT/'cases/native-q50/operator-spec.json')['boot_profiles'])
        for name in read_json(SUITE/'suite.json')['cases']:
            with self.subTest(case=name):
                directory = SUITE/name
                graph,case = read_json(directory/'graph.json'),read_json(directory/'case.json')
                self.assertTrue(200 <= len(graph['ops']) <= 800)
                self.assertEqual(len(graph['ops']),len(large.active_rows(graph)))
                recipe = read_json(directory/'recipe.json')
                plan,info = materialize_recipe(graph,case,spec,digest,recipe)
                self.assertEqual(hashlib.sha256(plan_bytes(plan)).hexdigest(),
                                 read_json(directory/'reference.json')['plan_sha256'])
                self.assertGreater(verify(plan,spec,case,digest)['boot_count'],0)
                for boundary in runner.plan_diagnostics(plan)['boot_boundaries']:
                    self.assertEqual((boundary['input_level'],boundary['output_level']),(5,30))
                    self.assertAlmostEqual(boundary['input_scale_log2'],40)
                    self.assertAlmostEqual(boundary['output_scale_log2'],40)
                # No refresh means the deep graph cannot meet the same contract.
                recipe['boots_before'] = {}
                with self.assertRaises((large.Infeasible,InvalidPlan)):
                    materialize_recipe(graph,case,spec,digest,recipe)

    def test_runner_checks_frozen_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            run = subprocess.run([sys.executable,str(ROOT/'run-references.py'),'--suite',str(SUITE),
                                  '--output-dir',temp],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertEqual(read_json(Path(temp)/'summary.json')['reference_count'],5)

    def test_bsgs_against_dense_matrix(self):
        g = large.Graph(); g.finish(large.linear(g.input('x'),seed=7,gain=.6))
        x = [.1*(i-3) for i in range(8)]
        actual = large.evaluate(g.data,[x])[0]
        expected = [sum(.6/8*(1+.2*math.sin((r+1)*(((c-r)%8)+1)+7))*x[c] for c in range(8)) for r in range(8)]
        for a,b in zip(actual,expected): self.assertAlmostEqual(a,b,places=13)

    def test_rotation_masks_epsilon_and_wrong_rotation(self):
        spec = read_json(SUITE/'operator-spec.json')
        digest = 'sha256:'+hashlib.sha256((SUITE/'operator-spec.json').read_bytes()).hexdigest()
        directory = SUITE/'qwen-attention-cached-2tokens'
        graph,case = read_json(directory/'graph.json'),read_json(directory/'case.json')
        source = runner.earth_source(graph,case,spec)
        self.assertIn('9.99999999999999955e-07',source)
        self.assertIn('offset = array<i64: -4>',source)
        self.assertIn('dense<[',source)
        plan = copy.deepcopy(reference_plan(directory))
        rotate = next(s for s in plan['execution'] if s.get('op')=='rotate')
        rotate['attrs']['steps'] = 2
        with self.assertRaisesRegex(InvalidPlan,'test vector'):
            verify(plan,spec,case,digest)


if __name__ == '__main__':
    unittest.main()
