#!/usr/bin/env python3
"""Verify profile-driven native Boot states and placement (not CKKS accuracy)."""
import argparse
import hashlib
import json
import runpy
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hecate-opt', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--boot-scale', type=int, choices=(40, 45), default=40)
    parser.add_argument('--strategy', choices=('depth-dp', 'optimized'), default='depth-dp')
    args = parser.parse_args()
    boot_scale = args.boot_scale
    boot_level = 27 if boot_scale == 40 else 22
    profile_id = f'native{boot_scale}'
    fixture_dir = args.source_dir / 'test/runtime-plan'
    source = runpy.run_path(str(fixture_dir / 'check-greedy.py'))['source']
    with tempfile.TemporaryDirectory(prefix='native-scale40-') as directory:
        temp = Path(directory)
        config = json.loads((args.source_dir / 'config.json').read_text())
        config.update(polynomialDegree=16, rescalingFactor=30,
                      bootstrapLevelLowerBound=1, bootstrapLevelUpperBound=boot_level-3,
                      levelLowerBound=1, levelUpperBound=boot_level-3)
        profile = temp / 'config.json'
        profile.write_text(json.dumps(config))
        spec = json.loads((fixture_dir / 'lazy-physical-level-operator-spec.json').read_text())
        # Enable ordinary operations for placement; the fixture latency is synthetic.
        for entry in spec['operators'].values():
            entry['supported'] = True
            entry['latency_us_by_level'] = [10] * 46
        spec['context']['default_scale_log2'] = 40
        spec['rescale_mode'] = 'native'
        spec['context']['rns_moduli'] = [(1 << 30) - (i + 1) * 32768 + 1 for i in range(50)]
        spec['boot_profiles'] = [dict(profile_id=profile_id, implementation='native',
            input_level_min=4, input_level_max=49, input_components=2,
            output_level=boot_level, output_scale_log2=boot_scale, output_components=2,
            latency_us_by_input_level=[85000] * 46, noise_by_input_level=None,
            host_requirements=dict(needs_secret_key=False, needs_host_compute=False))]
        spec_path = temp / 'spec.json'
        spec_path.write_text(json.dumps(spec))
        traced, output = temp / 'chain.mlir', temp / 'result.mlir'
        traced.write_text(source(40))
        for devices in (1, 4):
            command = [str(args.hecate_opt), '--dacapo', f'--boot-placement={args.strategy}',
                f'--ckks-config={profile}', '--waterline=40', f'--bootstrap-scale={boot_scale}',
                '--runtime-plan-format=json', f'--runtime-plan-boot-profile={profile_id}',
                '--runtime-plan-boot-implementation=native',
                f'--runtime-plan-physical-level-operator-spec-path={spec_path}',
                '--runtime-plan-levels-per-logical-level=1',
                f'--runtime-plan-device-counts={devices}', f'--runtime-plan-device-count={devices}',
                f'--runtime-plan-operator-spec-path={spec_path}',
                '--runtime-plan-target-id=lazy-physical-level-test',
                '--runtime-plan-context-id=lazy-physical-level-test',
                '--runtime-plan-operator-spec-id=lazy-physical-level-test-v1',
                '--runtime-plan-operator-spec-version=1',
                '--runtime-plan-operator-spec-sha256=sha256:' + hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                str(traced), '-o', str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            plan = json.loads(output.with_name('result.chain.runtime-plan.json').read_text())
            values = {v['id']: v for v in plan['values']}
            boots = [s for s in plan['execution'] if s.get('op') == 'boot']
            assert boots
            assert all(values[s['output']]['level'] == boot_level and
                       values[s['output']]['scale_log2'] == boot_scale and
                       s['place']['kind'] == 'device' for s in boots), boots
            assert max(v['level'] for v in plan['values']) == boot_level
            assert plan['format_version'] == 5
            assert all(0 <= v['scale_log2'] < 140 for v in plan['values'])
            # Boot consumes no extra rescale and requires no Host round trip.
            rescales = [s for s in plan['execution'] if s.get('op') == 'rescale']
            # Each of the 40 squares needs at least one RS. Optimized may
            # refresh before every square, leaving exactly 40 RS in the plan.
            assert 40 <= len(rescales) < 60, (args.strategy, boot_scale, len(rescales))
            assert all(values[s['inputs'][0]]['level'] - values[s['output']]['level'] == 1 for s in rescales)
            for phase in ('initialization', 'execution', 'finalization'):
                for step in plan[phase]:
                    if step.get('op') == 'boot':
                        assert 40 <= values[step['inputs'][0]]['scale_log2'] < 70.1
            invalid = command.copy()
            invalid[invalid.index(f'--runtime-plan-boot-profile={profile_id}')] = '--runtime-plan-boot-profile=missing'
            result = subprocess.run(invalid, capture_output=True, text=True)
            assert result.returncode != 0 and 'Boot profile is missing' in result.stderr
            invalid = [arg if arg != f'--bootstrap-scale={boot_scale}' else
                       f'--bootstrap-scale={45 if boot_scale == 40 else 40}' for arg in command]
            result = subprocess.run(invalid, capture_output=True, text=True)
            assert result.returncode != 0 and 'Boot result level/scale disagrees' in result.stderr
    print(f'Native Boot plans preserve level{boot_level}/scale{boot_scale} on 1 and 4 devices')


if __name__ == '__main__':
    main()
