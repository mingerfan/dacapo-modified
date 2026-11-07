#!/usr/bin/env python3
"""Compile scheduled Earth IR, then independently check exported physical plans."""
import argparse
from copy import deepcopy
import hashlib
import html
import json
import math
import re
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'test/scale-boot'))
from alignment import operation_counts, steps, verify_alignment_budget
from verify import read_json, verify


def earth(plan, nominal_rescale=30):
    """Keep actual encoding metadata; nominal types are parser scaffolding."""
    values = {v['id']: v for v in plan['values']}
    nominal = {v: math.ceil(values[v]['scale_log2']) for v in plan['external_inputs']}
    # A standalone pass initializes native parameters after parsing. The input
    # must initially type-check under the configured nominal rescale factor.
    for s in steps(plan):
        out = s['output']
        if s['kind'] == 'encode':
            nominal[out] = math.ceil(values[out]['scale_log2'])
        elif s['op'].startswith('mul'):
            nominal[out] = sum(nominal[v] for v in s['inputs'])
        elif s['op'] == 'rescale':
            nominal[out] = nominal[s['inputs'][0]] - nominal_rescale
        elif s['op'] == 'boot':
            nominal[out] = math.ceil(values[out]['scale_log2'])
        else:
            nominal[out] = nominal[s['inputs'][0]]
        if s.get('op') in ('add_cc', 'add_cp'):
            # Second operands here are function arguments or private constants.
            nominal[s['inputs'][1]] = nominal[out]
    def ty(v):
        d = values[v]
        kind = 'ci' if d['kind'] == 'ciphertext' else 'pl'
        return f'tensor<8x!earth.{kind}<{nominal[v]} * {30-d["level"]}>>'
    inputs = plan['external_inputs']
    outputs = plan['final_outputs']
    scales = ', '.join(repr(float(values[v]['scale_log2'])) for v in inputs)
    results = ', '.join(repr(float(values[v]['scale_log2'])) for v in outputs)
    lines = ['module {', 'func.func @graph(' + ', '.join(f'%v{v}: {ty(v)}' for v in inputs) +
             ') -> (' + ', '.join(ty(v) for v in outputs) + ') attributes {' +
             f'init_level = 30 : i64, earth.arg_scales = array<f64: {scales}>, ' +
             f'earth.result_scales = array<f64: {results}>' + '} {']
    for s in steps(plan):
        out = s['output']
        if s['kind'] == 'encode':
            payload = json.dumps(s['payload']['values'])
            lines.append(f'%v{out} = "earth.constant"() {{value = dense<{payload}> : tensor<8xf64>, '
                         f'rms_var = 1.0 : f64, earth.encoding_scale = {values[out]["scale_log2"]!r} : f64}} : () -> {ty(out)}')
            continue
        op = {'mul_cp':'mul', 'mul_cc':'mul', 'add_cc':'add', 'add_cp':'add', 'mod_switch':'modswitch',
              'boot':'bootstrap'}.get(s['op'], s['op'])
        attrs = ''
        if op == 'modswitch':
            drop = values[s['inputs'][0]]['level'] - values[out]['level']
            attrs = f' <{{downFactor = {drop} : i64}}>'
        if op == 'rotate':
            attrs = f' <{{offset = array<i64: {s["attrs"]["steps"]}>}}>'
        if op == 'bootstrap':
            attrs = f' <{{targetLevel = {30-values[out]["level"]} : i64, targetScale = {nominal[out]} : i64}}>'
        operands = ', '.join('%v'+v for v in s['inputs'])
        types = ', '.join(ty(v) for v in s['inputs'])
        lines.append(f'%v{out} = "earth.{op}"({operands}){attrs} : ({types}) -> {ty(out)}')
    lines += ['return ' + ', '.join('%v'+v for v in outputs) + ' : ' + ', '.join(ty(v) for v in outputs), '}', '}']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hecate-opt', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--graphs', action='store_true')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        run(args, args.output_dir.resolve() if args.output_dir else Path(temp))


def run(args, output):
    output.mkdir(parents=True, exist_ok=True)
    source = args.source_dir.resolve()
    suite = source / 'test/scale-boot/cases/alignment-q50'
    spec_path = suite / 'operator-spec.json'
    spec = read_json(spec_path)
    digest = 'sha256:' + hashlib.sha256(spec_path.read_bytes()).hexdigest()
    profile = spec['boot_profiles'][0]['profile_id']
    config = source / 'test/scale-boot/cases/model-large-q50/compiler.json'
    alignment = f'hoist-scale-alignment{{operator-spec-path={spec_path} boot-profile={profile} waterline=40}}'
    export = [f'--runtime-plan-physical-level-operator-spec-path={spec_path}',
              f'--runtime-plan-operator-spec-path={spec_path}', '--runtime-plan-levels-per-logical-level=1',
              f'--runtime-plan-target-id={spec["target_id"]}', f'--runtime-plan-context-id={spec["context"]["context_id"]}',
              f'--runtime-plan-operator-spec-id={spec["spec_id"]}', f'--runtime-plan-operator-spec-version={spec["version"]}',
              f'--runtime-plan-operator-spec-sha256={digest}', f'--runtime-plan-boot-profile={profile}',
              '--runtime-plan-format=json']
    report, diagrams = [], []
    for name in ('absorb-mulcp', 'reuse-modswitch', 'insufficient-budget', 'reuse-existing-rescale',
                 'addcc-without-unit', 'shared-boundary', 'fixed-boundary', 'shared-plaintext',
                 'rotate-addcp-negate', 'boot-boundaries', 'transaction-after-repair'):
        base = ('reuse-modswitch' if name == 'reuse-existing-rescale' else name
                if name in ('absorb-mulcp', 'insufficient-budget', 'reuse-modswitch') else 'absorb-mulcp')
        before = read_json(suite / base / 'before.json')
        case = read_json(suite / base / 'case.json')
        if name == 'addcc-without-unit':
            encode, mul = before['execution'][-3:-1]
            before['execution'][-1]['inputs'][0] = mul['inputs'][0]
            before['execution'][-3:-1] = []
            before['values'] = [v for v in before['values'] if v['id'] not in (encode['output'], mul['output'])]
        if name == 'shared-boundary':
            before['final_outputs'].append(before['execution'][-2]['inputs'][0])
        if name == 'transaction-after-repair':
            # First unit is repairable; the later independent one has no budget.
            # A failed pass must discard the earlier successful local rewrite.
            source_value = before['values'][0]
            plain = dict(deepcopy(source_value), id='late-plain', kind='plaintext', components=1, scale_log2=.2)
            out = dict(deepcopy(source_value), id='late-output', scale_log2=40.2)
            before['values'] += [plain, out]
            before['execution'] += [dict(kind='encode', output=plain['id'], payload=dict(kind='inline', values=[1.]*8)),
                dict(kind='compute', op='mul_cp', inputs=[source_value['id'], plain['id']], output=out['id'], place=out['place'])]
            before['final_outputs'].append(out['id'])
        if name == 'shared-plaintext':
            before['final_outputs'].append(before['execution'][0]['output'])
            case['outputs'].append(dict(case['outputs'][0], kind='plaintext', components=1,
                                        min_scale_log2=40, max_scale_log2=100))
            for vector in case['test_vectors']:
                vector['outputs'].append([.75]*8)
        if name == 'boot-boundaries':
            boot = spec['boot_profiles'][0]
            arg = dict(deepcopy(before['values'][0]), id='boot-input', level=5, scale_log2=40.)
            out = dict(deepcopy(arg), id='boot-output', level=boot['output_level'], scale_log2=boot['output_scale_log2'])
            before['values'] += [arg, out]
            before['external_inputs'].append(arg['id'])
            before['final_outputs'].append(out['id'])
            before['execution'].append(dict(ordinal=len(before['execution']), kind='compute', op='boot',
                inputs=[arg['id']], output=out['id'], place=arg['place'], attrs=dict(target_level=out['level'],
                target_scale_log2=out['scale_log2'], target_components=2, operator_profile=boot['profile_id'],
                implementation=boot['implementation'])))
            case['input_names'].append('boot-input')
            case['inputs'].append({k: arg[k] for k in ('kind', 'level', 'scale_log2', 'components', 'ntt')})
            case['outputs'].append(deepcopy(case['outputs'][0]))
            for vector in case['test_vectors']:
                vector['inputs'].append([.25]*8)
                vector['outputs'].append([.25]*8)
        if name == 'rotate-addcp-negate':
            # Extend the private chain before its inaccurate terminal unit.
            unit = before['execution'][-2]
            incoming = unit['inputs'][0]
            state = next(v for v in before['values'] if v['id'] == incoming)
            for name_id in ('rotation', 'bias', 'biased', 'negated'):
                row = dict(deepcopy(state), id=name_id)
                if name_id == 'bias':
                    row.update(kind='plaintext', components=1)
                before['values'].append(row)
            def compute(op, inputs, output_id, **attrs):
                return dict(kind='compute', op=op, inputs=inputs, output=output_id, place=state['place'], attrs=attrs)
            before['execution'][-3:-3] = [compute('rotate', [incoming], 'rotation', steps=2),
                dict(kind='encode', output='bias', payload=dict(kind='inline', values=[.3]*8)),
                compute('add_cp', ['rotation', 'bias'], 'biased'), compute('negate', ['biased'], 'negated')]
            unit['inputs'][0] = 'negated'
            for vector in case['test_vectors']:
                x, y = vector['inputs']
                vector['outputs'] = [[-(.75*a+.3)+b for a,b in zip(x[2:]+x[:2], y)]]
        if name == 'reuse-existing-rescale':
            rescale, encode, mul, add = before['execution']
            rescale['op'] = 'rescale'
            lost = math.log2(spec['context']['rns_moduli'][6])
            desc = {v['id']: v for v in before['values']}
            desc[rescale['output']]['scale_log2'] -= lost
            desc[encode['output']]['scale_log2'] += lost
            rescale['attrs']['target_scale_log2'] = desc[rescale['output']]['scale_log2']
        folder = output / name
        folder.mkdir(exist_ok=True)
        (folder / 'before.json').write_text(json.dumps(before, indent=2)+'\n')
        (folder / 'case.json').write_text(json.dumps(case, indent=2)+'\n')
        input_path = folder / 'input.mlir'
        text = earth(before)
        if name == 'fixed-boundary':
            target = before['execution'][-2]['inputs'][0]
            text = '\n'.join(line.replace(') : (', ') {sm_fixed} : (') if line.startswith(f'%v{target} =')
                             else line for line in text.splitlines())+'\n'
        input_path.write_text(text)
        parse_config = config
        if name == 'boot-boundaries':
            # Parsing precedes native-profile initialization in a standalone
            # pass. Permit the physical L5 Boot input in the nominal verifier.
            data = read_json(config)
            data['bootstrapLevelUpperBound'] = 30
            parse_config = folder/'compiler.json'
            parse_config.write_text(json.dumps(data, indent=2)+'\n')
        def compile_to(stem, passes, extra=()):
            cmd = [str(args.hecate_opt.resolve()), str(input_path), f'--ckks-config={parse_config}', *export,
                   '--pass-pipeline=builtin.module(' + passes + ')', '-o', str(folder / (stem+'.mlir')), *extra]
            result = subprocess.run(cmd, text=True, capture_output=True, timeout=60)
            (folder / (stem+'.log')).write_text(result.stdout+result.stderr)
            (folder / (stem+'.command.json')).write_text(json.dumps(cmd, indent=2)+'\n')
            return result
        result = compile_to('repaired', f'func.func({alignment})')
        if name in ('insufficient-budget', 'shared-boundary', 'fixed-boundary', 'transaction-after-repair'):
            assert result.returncode != 0 and 'hoist-scale-alignment:' in result.stderr, result.stderr
            assert 'no precision-safe candidate' in result.stderr, result.stderr
            failed = compile_to('failed-transaction', f'func.func({alignment})',
                                ('--mlir-disable-threading', '--mlir-print-ir-before-all', '--mlir-print-ir-after-failure'))
            assert failed.returncode != 0 and 'IR Dump After' in failed.stderr, failed.stderr
            original = failed.stderr.split('IR Dump After')[-1]
            assert 'earth.scale_alignment_policy' not in original
            assert 'earth.scale_alignment_trace' not in original
            def function_ir(dump):
                return dump.split('\n', 1)[1].split('\n\n', 1)[0]
            assert function_ir(failed.stderr.split('IR Dump Before')[-1]) == function_ir(original)
            report.append(dict(case=name, status='rejected', diagnostic=result.stderr))
            continue
        assert result.returncode == 0, result.stderr
        # The second invocation must keep exact encodings and boundary states.
        repeat = compile_to('idempotent', f'func.func({alignment},{alignment})')
        assert repeat.returncode == 0, repeat.stderr
        assert (folder/'repaired.mlir').read_text() == (folder/'idempotent.mlir').read_text()
        lowered = compile_to('compiled', f'func.func({alignment}),canonicalize,cse,'
                             'func.func(convert-earth-to-ckks,convert-upscale-to-mulcp),ckks-to-runtime-plan')
        assert lowered.returncode == 0, lowered.stderr
        automatic = read_json(folder/'compiled.graph.runtime-plan.json')
        stats = verify(automatic, spec, case, digest)
        contract = dict(format_version=1, boundaries=[dict(name=f'boundary-{i}', before=a, after=b)
            for i, (a,b) in enumerate(zip(before['external_inputs']+before['final_outputs'],
                                          automatic['external_inputs']+automatic['final_outputs']))])
        budget = verify_alignment_budget(before, automatic, contract)
        if name in ('absorb-mulcp', 'reuse-modswitch'):
            manual = read_json(suite/name/'candidate.json')
            manual_stats = verify(manual, spec, case, digest)
            assert stats['operation_counts'] == manual_stats['operation_counts']
            assert stats['encode_count'] == manual_stats['encode_count']
        if name == 'reuse-existing-rescale':
            assert budget['operation_delta']['rescale'] == 0
        if name == 'shared-plaintext':
            encoded = [s['payload']['values'] for s in steps(automatic) if s['kind']=='encode']
            assert encoded == [[.75]*8, [.75]*8], encoded
        trace = re.findall(r'strategy = "([^"]+)"', (folder/'repaired.mlir').read_text())
        report.append(dict(case=name, status='accepted', verification=stats, budget=budget,
                           strategies=trace, before=operation_counts(before), after=operation_counts(automatic)))
        diagrams.append((name, before, automatic, case))
    # Exercise the public pipeline switch, not only the explicit pass pipeline.
    # The threshold case needs absorption; regional cases also cover Boot plans
    # in the larger check-regional-scale.py CTests.
    for policy in ('threshold', 'regional'):
        for placement in ('greedy', 'optimized'):
            folder = output/f'pipeline-{policy}-{placement}'
            folder.mkdir(exist_ok=True)
            ct, pt = 'tensor<8x!earth.ci<0 * 0>>', 'tensor<8x!earth.pl<0 * 0>>'
            raw = (f'module {{func.func @graph(%x: {ct}, %y: {ct}) -> {ct} {{\n'
                   f'%p = "earth.constant"() {{value=dense<0.75>:tensor<8xf64>, rms_var=1.0:f64}} : () -> {pt}\n'
                   f'%a = "earth.mul"(%x, %p) : ({ct}, {pt}) -> {ct}\n'
                   f'%b = "earth.add"(%a, %y) : ({ct}, {ct}) -> {ct}\nreturn %b : {ct}\n}}}}\n')
            (folder/'input.mlir').write_text(raw)
            cmd = [str(args.hecate_opt.resolve()), str(folder/'input.mlir'), f'--ckks-config={config}', *export,
                   '--dacapo', f'--native-scale-policy={policy}', f'--boot-placement={placement}',
                   '--waterline=40', '--bootstrap-scale=40', '-o', str(folder/'compiled.mlir')]
            baseline = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            assert baseline.returncode == 0, baseline.stderr
            assert 'earth.scale_alignment_policy' not in (folder/'compiled.mlir').read_text()
            before = read_json(folder/'compiled.graph.runtime-plan.json')
            (folder/'before.json').write_text(json.dumps(before, indent=2)+'\n')
            cmd.append('--enable-scale-alignment-hoisting')
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, result.stderr
            (folder/'compiled.command.json').write_text(json.dumps(cmd, indent=2)+'\n')
            automatic = read_json(folder/'compiled.graph.runtime-plan.json')
            case = read_json(suite/'absorb-mulcp/case.json')
            for desc in case['inputs']:
                desc.update(level=30, scale_log2=40.)
            case['outputs'][0].update(min_level=4, max_scale_log2=100)
            stats = verify(automatic, spec, case, digest)
            contract = dict(format_version=1, boundaries=[dict(name=f'boundary-{i}', before=a, after=b)
                for i, (a,b) in enumerate(zip(before['external_inputs']+before['final_outputs'],
                                              automatic['external_inputs']+automatic['final_outputs']))])
            budget = verify_alignment_budget(before, automatic, contract)
            final = (folder/'compiled.mlir').read_text()
            assert 'earth.scale_alignment_policy' in final
            if policy == 'threshold':
                assert 'strategy = "absorb-mulcp"' in final
            report.append(dict(case=folder.name, status='accepted', verification=stats, budget=budget))
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    if args.graphs:
        from draw_graphs import DOT_STYLE, dot_body, plan_model
        dot = shutil.which('dot')
        if dot is None:
            raise RuntimeError('--graphs requires Graphviz dot')
        sections = []
        for name, before, after, case in diagrams:
            parts = ['digraph comparison {', DOT_STYLE]
            for label, plan in (('Before', before), ('Compiled', after)):
                body, _ = dot_body(plan_model(plan, case, spec, {}), label)
                parts += [f'subgraph cluster_{label} {{label="{label}";', body, '}']
            parts += ['}']
            path = output/name/'comparison.dot'
            path.write_text('\n'.join(parts))
            for ext in ('svg', 'png'):
                subprocess.run([dot, '-T'+ext, str(path), '-o', str(path.with_suffix('.'+ext))], check=True)
            sections.append(f'<h2>{html.escape(name)}</h2><img src="{name}/comparison.svg">')
        (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>独立 pass 验收</title>'
            '<style>body{font:16px system-ui;max-width:1400px;margin:32px auto}img{max-width:100%}pre{white-space:pre-wrap}</style>'
            '<h1>独立 pass → RuntimePlan 校验</h1><p>下图来自真实 hecate-opt 导出；所有接受方案均通过物理状态、'
            '固定边界、明文计算及全 1 量化校验。共享边界、固定边界和预算不足案例按预期拒绝。未执行加密误差测试。</p>'
            '<a href="summary.json">完整统计</a>'+''.join(sections))
    print(f'{len(report)} compiler cases passed; report: {output / "summary.json"}')


if __name__ == '__main__':
    main()
