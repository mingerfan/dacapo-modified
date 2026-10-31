"""Replay regional plans with real primes and fixed-scale Boot inputs.

Plaintext arithmetic treats Boot as an identity. This checks scheduling and
coefficient semantics, not encrypted Boot approximation error.
"""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import runpy
import subprocess
import tempfile


def replay(plan, graph, spec):
    desc = {v['id']: v for v in plan['values']}
    values = dict(zip(plan['external_inputs'], (.25, .75)))
    boots = 0
    for s in itertools.chain(plan['initialization'], plan['execution'], plan['finalization']):
        kind = s['kind']
        if kind == 'release':
            continue
        out = s['outputs'][0] if kind == 'transfer' else s.get('output')
        if kind in ('encode', 'prepare_plaintext'):
            payload = s['payload']['values']
            assert all(x == payload[0] for x in payload)
            values[out] = payload[0]
            assert desc[out]['scale_log2'] >= (30 if payload[0] == 1 else 40) - 1e-8, (out, payload[0], desc[out])
            continue
        inputs = s['inputs']
        before = [desc[x] for x in inputs]
        after = desc[out]
        op = s.get('op', kind)
        scale, level, value = before[0]['scale_log2'], before[0]['level'], values[inputs[0]]
        if op in ('mul_cc', 'mul_cp', 'add_cc', 'add_cp'):
            assert before[0]['level'] == before[1]['level']
            if op.startswith('mul'):
                scale += before[1]['scale_log2']
                value *= values[inputs[1]]
            else:
                assert abs(scale - before[1]['scale_log2']) < 1e-8
                value += values[inputs[1]]
        elif op == 'rescale':
            scale -= math.log2(spec['context']['rns_moduli'][level])
            level -= 1
        elif op == 'mod_switch':
            assert after['level'] < level
            level = after['level']
        elif op == 'boot':
            assert level == 5 and abs(scale - 40) < 1e-8
            level, scale = 30, 40
            boots += 1
        else:
            assert op in ('rotate', 'negate', 'relinearize', 'transfer', 'copy'), op
            if op == 'negate':
                value = -value
        assert level == after['level'] and abs(scale - after['scale_log2']) < 1e-8
        if after['kind'] == 'ciphertext':
            assert level >= spec['levels']['lower_bound']
            capacity = sum(math.log2(q) for q in spec['context']['rns_moduli'][:level+1])
            assert 40 - 1e-8 <= scale < capacity
        values[out] = value
    for output, expected in zip(plan['final_outputs'], graph.expected()):
        assert math.isclose(values[output], expected, rel_tol=1e-10, abs_tol=1e-10)
    return boots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hecate-opt', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    args = parser.parse_args()
    source = args.source_dir.resolve()
    definitions = runpy.run_path(str(source/'test/runtime-plan/check-threshold-graphs.py'))
    fixture = json.loads((source.parents[1]/'tests/fixtures/resnet20-manual-q50.json').read_text())
    spec = json.loads((source/'test/runtime-plan/lazy-physical-level-operator-spec.json').read_text())
    spec['rescale_mode'] = 'native'
    spec['context'].update(poly_degree=16, default_scale_log2=40,
                           rns_moduli=fixture['rns_moduli'],
                           rns_moduli_log2=[q.bit_length() for q in fixture['rns_moduli']])
    spec['levels'].update(lower_bound=4, upper_bound=49)
    for entry in spec['operators'].values():
        entry.update(supported=True, latency_us_by_level=[10]*50)
    spec['boot_profiles'] = [dict(profile_id='regional40', implementation='native',
        input_level_min=5, input_level_max=5, input_scale_log2=40, input_components=2,
        output_level=30, output_scale_log2=40, output_components=2,
        latency_us_by_input_level=[85000]*50, noise_by_input_level=None,
        host_requirements=dict(needs_secret_key=False, needs_host_compute=False))]
    config = json.loads((source/'config.json').read_text())
    config.update(polynomialDegree=16, rescalingFactor=30, bootstrapLevelLowerBound=1,
                  bootstrapLevelUpperBound=24, levelLowerBound=1, levelUpperBound=24)
    rows = definitions['cases']()
    g = definitions['Graph']()
    value = 'x'
    for _ in range(40):
        value = g.op('mul', value, 'p')
    g.outputs = [value]
    rows.append(('long-cp-chain', g, False))
    g = definitions['Graph']()
    value = g.op('rotate', 'x')
    for _ in range(3):
        value = g.op('mul', value, 'p')
    g.outputs = [value]
    rows.append(('terminal-linear', g, True))
    g = definitions['Graph']()
    g.outputs = [g.op('mul', g.op('rotate', 'x'), 'p')]
    rows.append(('terminal-q5', g, 'q7'))
    with tempfile.TemporaryDirectory(prefix='regional-scale-') as directory:
        root = Path(directory)
        (root/'spec.json').write_text(json.dumps(spec))
        (root/'config.json').write_text(json.dumps(config))
        for name, graph, exhausted in rows:
            exhausted = (25 if exhausted is True else 23 if exhausted == 'two-left'
                         else 24 if exhausted == 'q7' else 0)
            (root/'input.mlir').write_text(graph.source(exhausted).replace('tensor<1xf64>', 'tensor<8xf64>'))
            command = [str(args.hecate_opt.resolve()), str(root/'input.mlir'), '--dacapo',
                '--native-scale-policy=regional', '--boot-placement=greedy',
                f'--ckks-config={root}/config.json', '--waterline=40', '--bootstrap-scale=40',
                '--runtime-plan-format=json', '--runtime-plan-boot-profile=regional40',
                f'--runtime-plan-physical-level-operator-spec-path={root}/spec.json',
                f'--runtime-plan-operator-spec-path={root}/spec.json',
                '--runtime-plan-levels-per-logical-level=1',
                '--runtime-plan-target-id=lazy-physical-level-test',
                '--runtime-plan-context-id=lazy-physical-level-test',
                '--runtime-plan-operator-spec-id=lazy-physical-level-test-v1',
                '--runtime-plan-operator-spec-version=1',
                '--runtime-plan-operator-spec-sha256=sha256:'+hashlib.sha256((root/'spec.json').read_bytes()).hexdigest(),
                '-o', str(root/'result.mlir')]
            if name == 'terminal-q5':
                command.append('--regional-output-scale=40')
            result = subprocess.run(command, capture_output=True, text=True)
            assert result.returncode == 0, (name, result.stderr)
            plan = json.loads((root/'result.graph.runtime-plan.json').read_text())
            boots = replay(plan, graph, spec)
            if name == 'terminal-q5':
                assert boots == 0
                desc = {v['id']: v for v in plan['values']}
                output = desc[plan['final_outputs'][0]]
                assert output['level'] == 4 and abs(output['scale_log2'] - 40) < 1e-8
            if name == 'terminal-linear':
                assert boots == 0
                result = subprocess.run(command + ['--regional-output-scale=40'], capture_output=True, text=True)
                assert result.returncode == 0, result.stderr
                plan = json.loads((root/'result.graph.runtime-plan.json').read_text())
                assert replay(plan, graph, spec) == 1
                desc = {v['id']: v for v in plan['values']}
                assert abs(desc[plan['final_outputs'][0]]['scale_log2'] - 40) < 1e-8
    print(f'{len(rows)+1} regional plan replays passed with fixed Q6/s40 Boot input')


if __name__ == '__main__':
    main()
