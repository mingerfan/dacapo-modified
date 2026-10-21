#!/usr/bin/env python3
"""Check Boot boundary identity, repeated refreshes, shapes and contracts."""
import argparse
import json
from pathlib import Path
import re
import runpy
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hecate-opt', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='dacapo-optimized-') as temporary:
        root = Path(temporary)
        config = json.loads((args.source_dir / 'config.json').read_text())
        config.update(polynomialDegree=16, rescalingFactor=30,
                      bootstrapLevelLowerBound=1, bootstrapLevelUpperBound=19,
                      levelLowerBound=1, levelUpperBound=19)
        profile = root / 'config.json'
        profile.write_text(json.dumps(config))
        ct, pt = 'tensor<8x!earth.ci<0 * 0>>', 'tensor<8x!earth.pl<0 * 0>>'
        original = '\n'.join([
            f'module {{ func.func @scheduled(%x: {ct}, %y: {ct}) -> ({ct}, {ct}, {ct})',
            'attributes {btp_schedule = [array<i64: 4, 1>, array<i64: 5, 1>]} {',
            f'%p = "earth.constant"() {{value = dense<1.01> : tensor<8xf64>, rms_var = 1.01 : f64}} : () -> {pt}',
            *[f'%{name} = "earth.mul"(%x, %p) : ({ct}, {pt}) -> {ct}' for name in ['a', 'b', 'c']],
            f'return %a, %b, %c : {ct}, {ct}, {ct}', '}}'])
        source, output = root / 'input.mlir', root / 'output.mlir'
        def manage(text, boot_scale=45):
            source.write_text(text)
            return subprocess.run([
                str(args.hecate_opt), f'--ckks-config={profile}',
                '--pass-pipeline=builtin.module(func.func('
                'bootstrapping-placement{bootstrap-scale=45},'
                f'proactive-rescaling{{waterline=40 bootstrap-scale={boot_scale}}}))',
                str(source), '-o', str(output)], capture_output=True, text=True)
        result = manage(original)
        assert result.returncode == 0, result.stderr
        text = output.read_text()
        order = re.findall(r'"earth\.(mul|bootstrap)"', text)
        assert order == ['mul', 'bootstrap', 'mul', 'bootstrap', 'mul'], text
        boots = re.findall(r'(%\w+) = "earth.bootstrap"\((%\w+)\)', text)
        assert len(boots) == 2 and boots[1][1] == boots[0][0], text
        returns = re.search(r'func.func @scheduled.*? -> \((.*?)\)', text, re.DOTALL)[1]
        assert re.findall(r'!earth.ci<(\d+) \* (\d+)>', returns) == [
            ('50', '1'), ('55', '1'), ('55', '1')], text
        for changed, expected in [
            (original.replace('4, 1>, array<i64: 5, 1', '5, 1>, array<i64: 4, 1'), 'unordered Boot boundary'),
            (original.replace('4, 1>, array<i64: 5, 1', '0, 4>, array<i64: 5, 1'), 'invalid ciphertext target')]:
            result = manage(changed)
            assert result.returncode != 0 and expected in result.stderr, result.stderr
        result = manage(original, boot_scale=40)
        assert result.returncode != 0 and 'Boot output scale disagrees' in result.stderr, result.stderr

        # Original inputs and all segment frontier values must retain shape.
        # This deep graph needs multiple boundaries under a two-Q budget.
        config.update(bootstrapLevelUpperBound=3, levelUpperBound=3)
        profile.write_text(json.dumps(config))
        Graph = runpy.run_path(str(args.source_dir / 'test/runtime-plan/check-threshold-graphs.py'))['Graph']
        graph = Graph(); value = 'x'
        for _ in range(8): value = graph.op('mul', value, 'p')
        graph.outputs = [value, 'x']
        source.write_text(graph.source().replace('tensor<1x', 'tensor<8x'))
        result = subprocess.run([
            str(args.hecate_opt), '--dacapo-precompile', '--boot-placement=optimized',
            '--waterline=40', '--bootstrap-scale=45', f'--ckks-config={profile}',
            str(source), '-o', str(output)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        text = output.read_text()
        assert 'ckks.bootstrapc' in text and 'tensor<1x' not in text, text
        print('DaCapo preserves boundary timing, repeated value refreshes, 8-slot shapes and scale45 contracts')


if __name__ == '__main__':
    main()
