#!/usr/bin/env python3
"""Physical-plan acceptance tests for AddCC/MulCC/Square backtracking.

References are explicit recipes, never derived from the compiler's output.
"""
import argparse
from copy import deepcopy
import hashlib
import html
import json
import math
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'test/scale-boot'))
from alignment import operation_counts, steps, verify_alignment_budget
from alignment_example import make_case
from reference import Plan
from verify import InvalidPlan, read_json, verify

earth = runpy.run_path(str(ROOT/'test/runtime-plan/check-hoist-scale-alignment.py'))['earth']


def make_fixture(name, spec, digest, repaired=False):
    case = make_case('absorb-mulcp')
    case['case_id'] = name
    # Three independently chosen vector inputs, including signs/cancellation.
    case['input_names'] = ['x', 'y', 'z']
    case['inputs'] = [dict(case['inputs'][0], level=8) for _ in range(3)]
    case['outputs'][0].update(max_scale_log2=100)
    for vector in case['test_vectors']:
        vector['inputs'].append([.3, -.6, .9, -1.2, .4, -.8, 1.1, -.5])
    if name in ('mulcc-right', 'addcc-blocked-arm'):
        case['inputs'][0]['level'] = 6
    if name in ('square-unit-rescale', 'mulcc-unit-edge', 'mulcc-existing-rescale'):
        case['inputs'][0]['level'] = 6
    if name in ('mulcc-unit-edge','mulcc-existing-rescale'):
        case['inputs'][1]['level'] = 5
    if name == 'mulcc-existing-rescale':
        case['min_ciphertext_scale_log2'] = 0  # Build an intentionally invalid pre-hoist skeleton.
    if name == 'addcc-mixed-repair':
        case['inputs'][1]['level'] = 6
    if name == 'addcc-demand-square':
        case['inputs'][2].update(level=5, scale_log2=80-math.log2(spec['context']['rns_moduli'][6])+.2)
    if name == 'addcc-nominal-exit':
        case['inputs'][0].update(level=9, scale_log2=46.185804)
        case['inputs'][1].update(scale_log2=46.185804+math.log2(13_400_001))
    p = Plan(spec, case, digest)
    x, y, z = p.external
    q = spec['context']['rns_moduli']
    removed = math.log2(q[8])+math.log2(q[7])
    def cp(v, coefficient, encoding):
        a = p.state(v)
        plain = p.encode(coefficient, a['level'], encoding)
        return p.compute('mul_cp', [v, plain], a['level'], a['scale_log2']+encoding)
    def rs(v):
        a = p.state(v)
        s = a['scale_log2']-math.log2(q[a['level']])
        return p.compute('rescale', [v], a['level']-1, s, target_level=a['level']-1, target_scale_log2=s)
    def ms(v, level):
        return p.compute('mod_switch', [v], level, p.state(v)['scale_log2'], target_level=level)
    def weighted(v, coefficient, delta=0):
        return rs(rs(cp(v, coefficient, removed+delta)))
    def add(a, b):
        return p.compute('add_cc', [a,b], p.state(a)['level'], max(p.state(a)['scale_log2'],p.state(b)['scale_log2']))
    def mul(a, b):
        v = p.compute('mul_cc', [a,b], p.state(a)['level'], p.state(a)['scale_log2']+p.state(b)['scale_log2'], components=3)
        return p.compute('relinearize', [v], p.state(v)['level'], p.state(v)['scale_log2'])
    def rotate(v):
        return p.compute('rotate', [v], p.state(v)['level'], p.state(v)['scale_log2'], steps=2)
    def negate(v):
        return p.compute('negate', [v], p.state(v)['level'], p.state(v)['scale_log2'])
    extra = []
    delta = .2 if repaired else 0
    if name in ('addcc-two-arms', 'nested-addcc', 'addcc-blocked-arm'):
        a = x if name == 'addcc-blocked-arm' else weighted(x, .75, delta)
        b = weighted(y, .5, delta)
        out = add(b,a) if name == 'addcc-blocked-arm' else add(a,b)
        if name == 'nested-addcc':
            out = add(out, weighted(z, .3, delta))
    elif name == 'mulcc-both-arms':
        out = rs(mul(weighted(x,.75,delta), weighted(y,.5)))
    elif name == 'mulcc-right':
        out = rs(mul(x, weighted(y,.5,delta)))
    elif name in ('square', 'shared-square-blocked', 'addcc-demand-square'):
        a = weighted(x,.75, delta/2)
        out = rs(mul(a,a))
        if name == 'shared-square-blocked':
            extra.append(a)
        if name == 'addcc-demand-square':
            out = add(out,z)
    elif name in ('square-unit-rescale','mulcc-unit-edge'):
        split = .1 if name=='square-unit-rescale' else .2
        a = rs(cp(x,1.,math.log2(q[6])+split)) if repaired else ms(x,5)
        out = rs(mul(a,a if name=='square-unit-rescale' else y))
    elif name == 'addcc-mixed-repair':
        a = ms(weighted(x,.75,delta),5)
        b = rs(cp(y,1.,math.log2(q[6])+.2)) if repaired else ms(y,5)
        out = add(a,b)
    elif name == 'mulcc-existing-rescale':
        a = rs(cp(x,1.,math.log2(q[6])+.2)) if repaired else rs(x)
        out = mul(a,y)
    elif name == 'double-add':
        a = weighted(x,.75,delta)
        out = add(a,a)
    elif name == 'addcc-nominal-exit':
        # Old IR uses 47 + 23 = 70. After aligning to y, independently
        # rounding the new unit encoding would give 47 + 24 = 71.
        aligned = cp(ms(x,8), 1., math.log2(13_400_001) if repaired else 22.2)
        summed = add(aligned, y)
        scale = p.state(summed)['scale_log2']
        bias = p.encode(.5, 8, scale)
        out = p.compute('add_cp', [summed,bias], 8, scale)
    elif name in ('shared-two-edges', 'shared-no-budget'):
        a = weighted(x,.75)
        extra.append(a)  # Explicitly bind the shared prefix in the budget check.
        def branch(v):
            if name == 'shared-no-budget':
                return v
            return rs(cp(v, 1., math.log2(q[6])+.2)) if repaired else ms(v,5)
        out = add(rotate(branch(a)), negate(branch(a)))
    elif name == 'boot-blocked':
        a = ms(weighted(x,.75),5)
        p.bindings['preboot'] = a
        p.boot('preboot','fixed-boot')
        out = p.bindings['preboot']
        extra.extend([a, out])  # Bind the original Boot endpoints as well.
    else:
        raise ValueError(name)
    if not repaired and name not in ('addcc-demand-square', 'addcc-nominal-exit'):
        out = cp(out, 1., math.log2(q[6])+.2 if name=='mulcc-existing-rescale' else .2)
    p.bindings['out'] = out
    names = ['out']
    for i, value in enumerate(extra):
        p.bindings[f'extra{i}'] = value
        names.append(f'extra{i}')
        case['outputs'].append(deepcopy(case['outputs'][0]))
    before = p.export(names)
    case['min_ciphertext_scale_log2'] = 40
    # Expectations are the original arithmetic, independent of either schedule.
    for vector in case['test_vectors']:
        vx, vy, vz = vector['inputs']
        if name in ('addcc-two-arms', 'nested-addcc', 'addcc-blocked-arm'):
            weight = 1 if name == 'addcc-blocked-arm' else .75
            expected = [weight*a+.5*b+(.3*c if name=='nested-addcc' else 0) for a,b,c in zip(vx,vy,vz)]
        elif name == 'mulcc-both-arms':
            expected = [.75*a*.5*b for a,b in zip(vx,vy)]
        elif name == 'mulcc-right':
            expected = [a*.5*b for a,b in zip(vx,vy)]
        elif name in ('square','shared-square-blocked','addcc-demand-square'):
            expected = [(.75*a)**2+(c if name=='addcc-demand-square' else 0) for a,c in zip(vx,vz)]
        elif name == 'square-unit-rescale':
            expected = [a*a for a in vx]
        elif name in ('mulcc-unit-edge','mulcc-existing-rescale'):
            expected = [a*b for a,b in zip(vx,vy)]
        elif name == 'double-add':
            expected = [1.5*a for a in vx]
        elif name == 'addcc-nominal-exit':
            expected = [a+b+.5 for a,b in zip(vx,vy)]
        elif name == 'addcc-mixed-repair':
            expected = [.75*a+b for a,b in zip(vx,vy)]
        elif name in ('shared-two-edges','shared-no-budget'):
            expected = [.75*(a-b) for a,b in zip(vx[2:]+vx[:2],vx)]
        else:
            expected = [.75*a for a in vx]
        vector['outputs'] = [expected]+[[.75*a for a in vx] for _ in extra]
    return before, case


def earth_graph(plan):
    # Earth MulCC includes relinearization in lowering. Collapse only the explicit
    # RuntimePlan relinearize aliases when preparing Earth input.
    compact = deepcopy(plan)
    aliases, rows = {}, []
    for row in steps(compact):
        if 'inputs' in row:
            row['inputs'] = [aliases.get(v,v) for v in row['inputs']]
        if row.get('op') == 'relinearize':
            aliases[row['output']] = row['inputs'][0]
        else:
            rows.append(row)
    compact.update(initialization=[], execution=rows, finalization=[],
                   final_outputs=[aliases.get(v,v) for v in compact['final_outputs']])
    return earth(compact, nominal_rescale=32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hecate-opt', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--graphs', action='store_true')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        run(args,args.output_dir.resolve() if args.output_dir else Path(temp))


def run(args, output):
    output.mkdir(parents=True,exist_ok=True)
    source=args.source_dir.resolve()
    suite=source/'test/scale-boot/cases/alignment-q50'
    spec_path=suite/'operator-spec.json'
    spec=read_json(spec_path)
    digest='sha256:'+hashlib.sha256(spec_path.read_bytes()).hexdigest()
    config=read_json(source/'test/scale-boot/cases/model-large-q50/compiler.json')
    config['bootstrapLevelUpperBound']=30  # Standalone parsing before native setup.
    config['rescalingFactor']=32  # Nominal scaffolding for q5..q8; replay uses actual primes.
    (output/'compiler.json').write_text(json.dumps(config,indent=2)+'\n')
    alignment=f'hoist-scale-alignment{{operator-spec-path={spec_path} boot-profile={spec["boot_profiles"][0]["profile_id"]} waterline=40}}'
    flags=[f'--ckks-config={output}/compiler.json',f'--runtime-plan-operator-spec-path={spec_path}',
        f'--runtime-plan-physical-level-operator-spec-path={spec_path}','--runtime-plan-levels-per-logical-level=1',
        f'--runtime-plan-target-id={spec["target_id"]}',f'--runtime-plan-context-id={spec["context"]["context_id"]}',
        f'--runtime-plan-operator-spec-id={spec["spec_id"]}',f'--runtime-plan-operator-spec-version={spec["version"]}',
        f'--runtime-plan-operator-spec-sha256={digest}',f'--runtime-plan-boot-profile={spec["boot_profiles"][0]["profile_id"]}',
        '--runtime-plan-format=json']
    names=('addcc-two-arms','nested-addcc','mulcc-both-arms','mulcc-right','square','shared-two-edges',
           'addcc-demand-square','square-unit-rescale','mulcc-unit-edge','addcc-mixed-repair','mulcc-existing-rescale','double-add',
           'addcc-nominal-exit',
           'addcc-blocked-arm','shared-square-blocked','shared-no-budget','boot-blocked','query-limit')
    report,panels=[],[]
    for name in names:
        before,case=make_fixture('square' if name=='query-limit' else name,spec,digest)
        folder=output/name;folder.mkdir(exist_ok=True)
        for stem,data in (('before',before),('case',case)):
            (folder/(stem+'.json')).write_text(json.dumps(data,indent=2)+'\n')
        (folder/'input.mlir').write_text(earth_graph(before))
        selected=alignment.replace('waterline=40','waterline=40 max-graph-queries=1') if name=='query-limit' else alignment
        def compile_to(stem,passes,extra=()):
            cmd=[str(args.hecate_opt.resolve()),str(folder/'input.mlir'),*flags,
                 '--pass-pipeline=builtin.module('+passes+')','-o',str(folder/(stem+'.mlir')),*extra]
            result=subprocess.run(cmd,text=True,capture_output=True,timeout=60)
            (folder/(stem+'.log')).write_text(result.stdout+result.stderr)
            (folder/(stem+'.command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
            return result
        result=compile_to('repaired',f'func.func({selected})')
        rejected=name in ('addcc-blocked-arm','shared-square-blocked','shared-no-budget','boot-blocked','query-limit')
        if rejected:
            expected='graph search limit exceeded' if name=='query-limit' else 'no precision-safe candidate'
            assert result.returncode!=0 and expected in result.stderr,(name,result.stderr)
            failed=compile_to('rollback',f'func.func({selected})',
                ('--mlir-disable-threading','--mlir-print-ir-before-all','--mlir-print-ir-after-failure'))
            def body(marker):
                return failed.stderr.split(marker)[-1].split('\n',1)[1].split('\n\n',1)[0]
            assert 'IR Dump After' in failed.stderr and body('IR Dump Before')==body('IR Dump After'),failed.stderr
            report.append(dict(case=name,status='rejected',diagnostic=expected))
            continue
        assert result.returncode==0,(name,result.stderr)
        if name=='addcc-nominal-exit':
            repaired_ir = (folder/'repaired.mlir').read_text()
            assert '(tensor<8x!earth.ci<70 * 22>>, tensor<8x!earth.pl<70 * 22>>) -> tensor<8x!earth.ci<70 * 22>>' in repaired_ir
        repeat=compile_to('idempotent',f'func.func({alignment},{alignment})')
        assert repeat.returncode==0,repeat.stderr
        assert (folder/'repaired.mlir').read_text()==(folder/'idempotent.mlir').read_text()
        lowered=compile_to('compiled',f'func.func({alignment}),canonicalize,cse,'
            'func.func(convert-earth-to-ckks,convert-upscale-to-mulcp),ckks-to-runtime-plan')
        assert lowered.returncode==0,(name,lowered.stderr)
        automatic=read_json(folder/'compiled.graph.runtime-plan.json')
        stats=verify(automatic,spec,case,digest)
        manual,_=make_fixture(name,spec,digest,True)
        manual_stats=verify(manual,spec,case,digest)
        (folder/'manual.json').write_text(json.dumps(manual,indent=2)+'\n')
        contract=dict(format_version=1,boundaries=[dict(name=f'boundary-{i}',before=a,after=b)
            for i,(a,b) in enumerate(zip(before['external_inputs']+before['final_outputs'],automatic['external_inputs']+automatic['final_outputs']))])
        budget=verify_alignment_budget(before,automatic,contract)
        (folder/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
        # Also bind the manual recipe against the same fixed exit and shared values.
        manual_contract=deepcopy(contract)
        for binding,value in zip(manual_contract['boundaries'],manual['external_inputs']+manual['final_outputs']):
            binding['after']=value
        verify_alignment_budget(before,manual,manual_contract)
        (folder/'manual-contract.json').write_text(json.dumps(manual_contract,indent=2)+'\n')
        # Downstream canonicalization can share RS after an addition and CSE
        # equal edge conversions. It must not exceed the independent recipe.
        assert all(count<=manual_stats['operation_counts'][op]
                   for op,count in stats['operation_counts'].items()),(name,stats,manual_stats)
        assert stats['encode_count']<=manual_stats['encode_count'],(name,stats,manual_stats)
        if name=='square':
            assert stats['operation_counts']['mul_cp']==1
            mul=next(s for s in steps(automatic) if s.get('op')=='mul_cc')
            assert mul['inputs'][0]==mul['inputs'][1],mul
            enc=next(v for v in automatic['values'] if v['kind']=='plaintext')
            expected=math.log2(spec['context']['rns_moduli'][8])+math.log2(spec['context']['rns_moduli'][7])+.1
            assert abs(enc['scale_log2']-expected)<1e-8,enc
        row=dict(case=name,status='accepted',verification=stats,budget=budget,
                 before=operation_counts(before),manual=operation_counts(manual),after=operation_counts(automatic),
                 strategies=re.findall(r'strategy = "([^"]+)"',(folder/'repaired.mlir').read_text()))
        report.append(row);panels.append((name,before,manual,automatic,case))
    # Public pipeline integration: the requested alignment is after a square,
    # and can only be absorbed in the MulCP before that square.
    for placement in ('greedy','optimized'):
        folder=output/f'pipeline-square-{placement}';folder.mkdir(exist_ok=True)
        ct,pt='tensor<8x!earth.ci<0 * 0>>','tensor<8x!earth.pl<0 * 0>>'
        raw=(f'module {{func.func @graph(%x: {ct}, %y: {ct}, %z: {ct}) -> {ct} {{\n'
             f'%p = "earth.constant"() {{value=dense<0.75>:tensor<8xf64>, rms_var=1.0:f64}} : () -> {pt}\n'
             f'%a = "earth.mul"(%x, %p) : ({ct}, {pt}) -> {ct}\n'
             f'%s = "earth.mul"(%a, %a) : ({ct}, {ct}) -> {ct}\n'
             f'%out = "earth.add"(%s, %y) : ({ct}, {ct}) -> {ct}\nreturn %out : {ct}\n}}}}\n')
        (folder/'input.mlir').write_text(raw)
        case=make_fixture('square',spec,digest)[1]
        for desc in case['inputs']:
            desc.update(level=30,scale_log2=40.)
        for vector in case['test_vectors']:
            x,y,_=vector['inputs']
            vector['outputs']=[[a*a*.75*.75+b for a,b in zip(x,y)]]
        (folder/'case.json').write_text(json.dumps(case,indent=2)+'\n')
        cmd=[str(args.hecate_opt.resolve()),str(folder/'input.mlir'),*flags,'--dacapo',
             '--native-scale-policy=threshold',f'--boot-placement={placement}','--waterline=40','--bootstrap-scale=40']
        baseline=subprocess.run([*cmd,'-o',str(folder/'baseline.mlir')],text=True,capture_output=True,timeout=60)
        assert baseline.returncode==0,baseline.stderr
        before=read_json(folder/'baseline.graph.runtime-plan.json')
        try:
            verify(before,spec,case,digest)
        except InvalidPlan as error:
            assert 'unit encoding error' in str(error),str(error)
            baseline_error=str(error)
        else:
            raise AssertionError('fixture must expose the baseline unit encoding error')
        cmd+=['--enable-scale-alignment-hoisting','-o',str(folder/'compiled.mlir')]
        result=subprocess.run(cmd,text=True,capture_output=True,timeout=60)
        (folder/'compiled.log').write_text(result.stdout+result.stderr)
        (folder/'compiled.command.json').write_text(json.dumps(cmd,indent=2)+'\n')
        assert result.returncode==0,result.stderr
        automatic=read_json(folder/'compiled.graph.runtime-plan.json')
        stats=verify(automatic,spec,case,digest)
        contract=dict(format_version=1,boundaries=[dict(name=f'boundary-{i}',before=a,after=b)
            for i,(a,b) in enumerate(zip(before['external_inputs']+before['final_outputs'],automatic['external_inputs']+automatic['final_outputs']))])
        budget=verify_alignment_budget(before,automatic,contract)
        (folder/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
        trace=re.findall(r'strategy = "([^"]+)"',(folder/'compiled.mlir').read_text())
        assert any(s.startswith('graph-') for s in trace),trace
        report.append(dict(case=folder.name,status='accepted',verification=stats,budget=budget,
                           baseline_rejected=baseline_error,strategies=trace))
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    if args.graphs:
        from draw_graphs import DOT_STYLE,dot_body,plan_model
        dot=shutil.which('dot')
        if not dot: raise RuntimeError('--graphs requires Graphviz dot')
        sections=[]
        for name,before,manual,automatic,case in panels:
            lines=['digraph comparison {',DOT_STYLE]
            for label,plan in (('Before',before),('Manual',manual),('Compiled',automatic)):
                body,_=dot_body(plan_model(plan,case,spec,{}),label)
                lines += [f'subgraph cluster_{label} {{label="{label}";',body,'}']
            path=output/name/'comparison.dot';path.write_text('\n'.join(lines+['}']))
            for ext in ('svg','png'):
                subprocess.run([dot,'-T'+ext,str(path),'-o',str(path.with_suffix('.'+ext))],check=True,timeout=60)
            sections.append(f'<h2>{html.escape(name)}</h2><p>'
                f'<a href="{name}/input.mlir">输入 MLIR</a> · <a href="{name}/repaired.mlir">改写后 MLIR</a> · '
                f'<a href="{name}/manual.json">人工方案</a> · <a href="{name}/compiled.graph.runtime-plan.json">编译结果</a></p>'
                f'<a href="{name}/comparison.svg"><img src="{name}/comparison.svg"></a>')
        rows=[]
        for row in report:
            name=html.escape(row['case'])
            detail=(f'Boot={row["verification"]["boot_count"]}，固定边界 level/scale 不变'
                    if row['status']=='accepted' else html.escape(row['diagnostic']))
            rows.append(f'<tr><td><a href="{name}/input.mlir">{name}</a></td><td>{row["status"]}</td><td>{detail}</td></tr>')
        (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>多输入 scale 回溯验收</title>'
            '<style>body{font:16px system-ui;margin:32px}img{max-width:100%}td,th{padding:8px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style><h1>计算图回溯：人工方案与实际编译结果</h1>'
            '<p>固定边界 level/scale 和 Boot 不变；人工方案与编译结果分别通过物理状态、明文及全 1 量化重放。'
            '未执行真实加密。点击图可打开 SVG 查看细节。</p><a href="summary.json">完整统计与拒绝案例</a>'
            '<table><tr><th>案例</th><th>结果</th><th>检查</th></tr>'+''.join(rows)+'</table>'+''.join(sections))
    print(f'{len(report)} graph compiler cases passed; report: {output/"summary.json"}')


if __name__=='__main__':
    main()
