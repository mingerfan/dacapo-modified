#!/usr/bin/env python3
"""Check scale absorption, full-slot identities, masks, and Encode sharing."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def tensor(kind, scale, level=0):
    return f'tensor<8x!earth.{kind}<{scale} * {level}>>'


def constant(name, scale, length=8, coefficient=0.5):
    return (f'%{name} = "earth.constant"() {{value = dense<{coefficient}> : '
            f'tensor<{length}xf64>, rms_var = 1.0 : f64}} : () -> {tensor("pl", scale)}')


def unary(name, op, source, source_type, result_type, factor=None):
    attr = f' {{upFactor = {factor} : i64}}' if factor is not None else ''
    return f'%{name} = "earth.{op}"(%{source}){attr} : ({source_type}) -> {result_type}'


def product(name, left, right, left_type, right_type, result_type):
    return f'%{name} = "earth.mul"(%{left}, %{right}) : ({left_type}, {right_type}) -> {result_type}'


def fixture(lines, outputs):
    results = ', '.join(ty for _, ty in outputs)
    names = ', '.join('%' + name for name, _ in outputs)
    return (f'module {{ func.func @test(%x: {tensor("ci", 40)}, %y: {tensor("ci", 40)}) '
            f'-> ({results}) attributes {{init_level = 5 : i64}} {{\n' +
            '\n'.join(lines) + f'\nreturn {names} : {results}\n}}}}\n')


def execute(plan):
    """Track scaled coefficients so incorrect scale rewrites affect the result."""
    types = {v['id']: v for v in plan['values']}
    coefficients = {}
    for index, value in enumerate(plan['external_inputs']):
        coefficients[value] = [(slot + 1 + index) / 8 * 2**types[value]['scale_log2']
                               for slot in range(8)]
    for step in plan['initialization']:
        output = step['output']
        payload = step['payload']['values']
        coefficients[output] = [x * 2**types[output]['scale_log2']
                                for x in payload + [0] * (8 - len(payload))]
    for step in plan['execution']:
        inputs = step['inputs']
        output = step['output']
        kind = step['op']
        if kind.startswith('mul_'):
            assert types[output]['scale_log2'] == sum(types[x]['scale_log2'] for x in inputs)
            assert types[inputs[0]]['level'] == types[inputs[1]]['level'] == types[output]['level']
            result = [a * b for a, b in zip(*(coefficients[x] for x in inputs))]
        elif kind == 'rescale':
            drop = types[inputs[0]]['scale_log2'] - types[output]['scale_log2']
            assert drop == 120
            result = [x / 2**drop for x in coefficients[inputs[0]]]
        elif kind == 'mod_switch':
            assert types[inputs[0]]['scale_log2'] == types[output]['scale_log2']
            result = coefficients[inputs[0]]
        elif kind == 'negate':
            result = [-x for x in coefficients[inputs[0]]]
        else:
            assert kind == 'relinearize', kind
            result = coefficients[inputs[0]]
        coefficients[output] = result
    return [[x / 2**types[value]['scale_log2'] for x in coefficients[value]]
            for value in plan['final_outputs']]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hecate-opt', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    args = parser.parse_args()
    ci40, ci80, ci160 = (tensor('ci', n) for n in (40, 80, 160))
    pl40 = tensor('pl', 40)
    cases = []
    for length, coefficient, swap in ((8, 0.5, False), (3, 1.0, False),
                                      (3, -1.0, True), (8, 0.5, True),
                                      (8, 1.0, False), (8, -1.0, False)):
        left, right, lt, rt = ('p', 'x', pl40, ci40) if swap else ('x', 'p', ci40, pl40)
        lines = [constant('p', 40, length, coefficient), product('m', left, right, lt, rt, ci80),
                 unary('u', 'upscale', 'm', ci80, ci160, 80),
                 unary('r', 'rescale', 'u', ci160, tensor('ci', 40, 1))]
        cases.append((fixture(lines, [('r', tensor('ci', 40, 1))]),
                      0 if length == 8 and abs(coefficient) == 1 else 1,
                      0 if length == 8 and abs(coefficient) == 1 else 1))
    # A shared product is observed at two scales, so absorption must not change it.
    lines = [constant('p', 40), product('m', 'x', 'p', ci40, pl40, ci80),
             unary('u', 'upscale', 'm', ci80, ci160, 80)]
    cases.append((fixture(lines, [('m', ci80), ('u', ci160)]), 2, 2))
    # A shared plaintext must keep the original encoding for its other user.
    lines = [constant('p', 40), product('m', 'x', 'p', ci40, pl40, ci80),
             unary('u', 'upscale', 'm', ci80, ci160, 80)]
    cases.append((fixture(lines, [('p', pl40), ('u', ci160)]), 1, 2))
    # Ciphertext products have no plaintext into which the scale can be folded.
    lines = [product('m', 'x', 'y', ci40, ci40, ci80),
             unary('u', 'upscale', 'm', ci80, ci160, 80)]
    cases.append((fixture(lines, [('u', ci160)]), 1, 1))
    # Necessary equal units can share one Encode; different scales cannot.
    for second_scale, expected_encodes in ((80, 1), (120, 2)):
        lines = [unary('a', 'upscale', 'x', ci40, ci80, 40),
                 unary('b', 'upscale', 'y', ci40, tensor('ci', second_scale), second_scale - 40)]
        cases.append((fixture(lines, [('a', ci80), ('b', tensor('ci', second_scale))]), 2, expected_encodes))
    lines = [unary('a', 'upscale', 'x', ci40, ci80, 40),
             f'%d = "earth.modswitch"(%y) {{downFactor = 1 : i64}} : ({ci40}) -> {tensor("ci", 40, 1)}',
             unary('b', 'upscale', 'd', tensor('ci', 40, 1), tensor('ci', 80, 1), 40)]
    cases.append((fixture(lines, [('a', ci80), ('b', tensor('ci', 80, 1))]), 2, 2))
    lines = [constant('p', 0, coefficient=1.0),
             product('m', 'x', 'p', ci40, tensor('pl', 0), ci40)]
    cases.append((fixture(lines, [('m', ci40)]), 0, 0))

    with tempfile.TemporaryDirectory(prefix='dacapo-unit-plaintext-') as directory:
        temp = Path(directory)
        config = json.loads((args.source_dir / 'config.json').read_text())
        config.update(polynomialDegree=16, rescalingFactor=120,
                      levelUpperBound=5, bootstrapLevelUpperBound=5)
        profile = temp / 'profile.json'
        profile.write_text(json.dumps(config))
        for index, (source, expected_muls, expected_encodes) in enumerate(cases):
            path = temp / 'input.mlir'
            path.write_text(source)
            plans = []
            for optimized in (False, True):
                prefix = temp / f'case-{index}-{optimized}'
                passes = ['canonicalize', 'cse'] if optimized else []
                passes += ['convert-earth-to-ckks', 'convert-upscale-to-mulcp']
                if optimized:
                    passes += ['cse', 'canonicalize']
                passes += ['emit-runtime-plan{' + f'prefix={prefix} plan-id=1 target-id=test '
                           'capability-version=1 operator-spec-id=test operator-spec-version=1 '
                           'operator-spec-sha256=sha256:' + '0'*64 +
                           ' context-id=test device-count=0 ntt=true}']
                command = [str(args.hecate_opt), str(path), f'--ckks-config={profile}',
                           '-p=builtin.module(func.func(' + ','.join(passes) + '))',
                           '-o', str(temp / 'output.mlir')]
                result = subprocess.run(command, capture_output=True, text=True)
                assert result.returncode == 0, (index, optimized, result.stderr)
                plans.append(json.loads(Path(f'{prefix}.test.runtime-plan.json').read_text()))
            assert execute(plans[0]) == execute(plans[1]), index
            assert sum(s['op'] == 'mul_cp' for s in plans[1]['execution']) == expected_muls, index
            assert len(plans[1]['initialization']) == expected_encodes, index
            before = {v['id']: v for v in plans[0]['values']}
            after = {v['id']: v for v in plans[1]['values']}
            for left, right in zip(plans[0]['final_outputs'], plans[1]['final_outputs']):
                for key in ('kind', 'level', 'scale_log2', 'components'):
                    assert before[left][key] == after[right][key], (index, key)
    print(f'{len(cases)} unit plaintext cases passed: scale, masks, sharing, and decoded values')


if __name__ == '__main__':
    main()
