#!/usr/bin/env python3
"""Check actual-prime scale/Boot planning and mixed-width physical replay.

No GPU, encryption, or CKKS accuracy claims are involved.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile


def source(depth, residual=False):
    ct = 'tensor<1x!earth.ci<0 * 0>>'
    lines = [f'module {{ func.func @squares(%x: {ct}) -> {ct} {{']
    previous = '%x'
    for i in range(depth):
        lines.append(f'%v{i} = "earth.mul"({previous}, {previous}) : ({ct}, {ct}) -> {ct}')
        previous = f'%v{i}'
    if residual:
        lines.append(f'%sum = "earth.add"({previous}, %x) : ({ct}, {ct}) -> {ct}')
        previous = '%sum'
    return '\n'.join(lines + [f'return {previous} : {ct}', '}}'])


def normalize(scale, consumed, moduli, anchor):
    while consumed <= anchor and scale - math.log2(moduli[anchor-consumed]) >= 40:
        scale -= math.log2(moduli[anchor-consumed])
        consumed += 1
    return scale, consumed


def minimum_boots(depth, moduli, anchor, budget):
    costs = []
    for cuts in itertools.product((False, True), repeat=depth):
        scale, consumed, boots = 40., 0, 0
        for cut in cuts:
            if cut:
                scale, consumed, boots = 40., 0, boots+1
            scale, consumed = normalize(2*scale, consumed, moduli, anchor)
            if consumed > budget:
                break
        else:
            costs.append(boots)
    return min(costs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hecate-opt', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    args = parser.parse_args()
    fixture = args.source_dir / 'test/runtime-plan/lazy-physical-level-operator-spec.json'
    config_source = json.loads((args.source_dir/'config.json').read_text())
    checks = 0
    with tempfile.TemporaryDirectory(prefix='native-planning-') as directory:
        out = Path(directory)

        def compile_case(moduli, depth, budget, strategy, residual=False):
            anchor = len(moduli)-1
            config = dict(config_source, polynomialDegree=16, rescalingFactor=30,
                          bootstrapLevelLowerBound=1, bootstrapLevelUpperBound=budget+1,
                          levelLowerBound=1, levelUpperBound=budget+1)
            spec = json.loads(fixture.read_text())
            spec['rescale_mode'] = 'native'
            spec['context']['rns_moduli'] = moduli
            spec['context']['rns_moduli_log2'] = [q.bit_length() for q in moduli]
            spec['levels'] = dict(lower_bound=anchor-budget, upper_bound=anchor)
            spec['boot_profiles'] = [dict(profile_id='actual40', output_level=anchor,
                                          output_scale_log2=40)]
            (out/'config.json').write_text(json.dumps(config))
            (out/'spec.json').write_text(json.dumps(spec))
            (out/'input.mlir').write_text(source(depth, residual))
            command = [str(args.hecate_opt.resolve()), '--dacapo-precompile',
                       f'--boot-placement={strategy}', f'--ckks-config={out}/config.json',
                       '--waterline=40', '--bootstrap-scale=40', '--mlir-disable-threading',
                       f'--runtime-plan-physical-level-operator-spec-path={out}/spec.json',
                       '--runtime-plan-boot-profile=actual40', str(out/'input.mlir'),
                       '-o', str(out/'result.mlir')]
            result = subprocess.run(command, text=True, capture_output=True, timeout=60)
            assert result.returncode == 0, (strategy, result.stderr[-6000:])
            text = (out/'result.mlir').read_text()
            scales = re.search(r'earth.result_scales = array<f64: ([^>]+)>', text)
            assert scales, text
            return text, float(scales[1])

        # Same source, different bit widths: two squares consume 3 versus 2 Q.
        uniform = [(1 << 30)-131072*i-1 for i in range(12)]
        mixed = [(1 << 32)-131072*i-1 for i in range(12)]
        mixed[0] = uniform[0]
        for strategy in ('greedy', 'depth-dp', 'optimized'):
            for chain, count in ((uniform, 3), (mixed, 2)):
                text, actual = compile_case(chain, 2, 8, strategy)
                expected, level = 40., 0
                for _ in range(2):
                    expected, level = normalize(2*expected, level, chain, len(chain)-1)
                assert level == count
                assert text.count('"ckks.rescalec"') == count, text
                assert math.isclose(actual, expected, abs_tol=1e-9), (actual, expected)
                assert '"ckks.bootstrapc"' not in text
                checks += 1

        # Equal nominal bit width is insufficient: these 30-bit moduli are
        # near 2^29, and five squares consume more Q than the near-2^30 chain.
        low = [(1 << 29)+131072*i+1 for i in range(16)]
        high = [(1 << 30)-131072*i-1 for i in range(16)]
        for chain in (low, high):
            text, actual = compile_case(chain, 5, 12, 'greedy')
            expected, level = 40., 0
            for _ in range(5):
                expected, level = normalize(2*expected, level, chain, len(chain)-1)
            assert text.count('"ckks.rescalec"') == level
            assert math.isclose(actual, expected, abs_tol=1e-9)
            checks += 1

        # Check DP and optimized candidate feasibility against independent
        # exhaustive cut enumeration, including non-integer frontier scales.
        for chain in (uniform, mixed):
            expected = minimum_boots(5, chain, len(chain)-1, 3)
            for strategy in ('depth-dp', 'optimized'):
                text, _ = compile_case(chain, 5, 3, strategy)
                assert text.count('"ckks.bootstrapc"') == expected, (strategy, expected, text)
                checks += 1
            for strategy in ('greedy', 'depth-dp', 'optimized'):
                text, actual = compile_case(chain, 3, 8, strategy, residual=True)
                assert math.isfinite(actual) and actual >= 40
                assert '"ckks.addcc"' in text
                checks += 1

        def physical():
            pipeline = ('builtin.module(func.func(materialize-ckks-physical-levels{'
                        f'operator-spec={out}/spec.json boot-profile=actual40 '
                        'levels-per-logical-level=1}))')
            return subprocess.run([str(args.hecate_opt.resolve()),
                f'--ckks-config={out}/config.json', f'--pass-pipeline={pipeline}',
                str(out/'result.mlir'), '-o', str(out/'physical.mlir')],
                text=True, capture_output=True, timeout=60)

        # Checkpoint replay validates exact primes, the Boot anchor and actual
        # output scales. Changing a checkpoint's context must fail explicitly.
        text, _ = compile_case(uniform, 2, 8, 'greedy')
        result = physical()
        assert result.returncode == 0, result.stderr
        checks += 1
        spec_text = (out/'spec.json').read_text()
        spec = json.loads(spec_text); spec['context']['rns_moduli'][0] -= 2
        (out/'spec.json').write_text(json.dumps(spec))
        result = physical()
        assert result.returncode != 0 and 'chain disagrees' in result.stderr, result.stderr
        checks += 1
        spec = json.loads(spec_text); spec['boot_profiles'][0]['output_level'] -= 1
        (out/'spec.json').write_text(json.dumps(spec))
        result = physical()
        assert result.returncode != 0 and 'Boot anchor disagrees' in result.stderr, result.stderr
        checks += 1
        (out/'spec.json').write_text(spec_text)
        bad = re.sub(r'(earth.result_scales = array<f64: )[^>]+', r'\g<1>99.0', text)
        (out/'result.mlir').write_text(bad)
        result = physical()
        assert result.returncode != 0 and 'planned result scale disagrees' in result.stderr, result.stderr
        checks += 1
        (out/'result.mlir').write_text(text.replace('earth.scale_waterline = 4.000000e+01',
                                                 'earth.scale_waterline = 7.000000e+01'))
        result = physical()
        assert result.returncode != 0 and 'scale waterline' in result.stderr, result.stderr
        checks += 1
        compile_case(mixed, 2, 8, 'greedy')
        result = physical()
        assert result.returncode == 0, result.stderr
        checks += 1
        spec = json.loads((out/'spec.json').read_text())
        spec['context']['rns_moduli_log2'][0] += 1
        (out/'spec.json').write_text(json.dumps(spec))
        result = physical()
        assert result.returncode != 0 and 'declared bit width' in result.stderr, result.stderr
        checks += 1
    print(f'{checks} actual-prime planning cases passed')


if __name__ == '__main__':
    main()
