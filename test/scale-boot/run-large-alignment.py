#!/usr/bin/env python3
"""Compare hoisting off/on on frozen model-pattern graphs, without a GPU.

Every export is checked under BOTH the frozen scale-floor contract and the
unit-quantization contract used by the alignment pass. No frozen fixture changes.
A nonzero exit means at least one compiled variant failed or was invalid.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import html
import json
import math
from pathlib import Path
import runpy
import subprocess
import time

from alignment import steps, verify_alignment_budget
from verify import InvalidPlan, decoded_unit, read_json, verify
from reference import reference_plan

ROOT = Path(__file__).resolve().parent
helpers = runpy.run_path(str(ROOT/'run-references.py'))


def check(plan, spec, case, digest):
    try:
        return dict(valid=True, stats=verify(plan,spec,case,digest))
    except InvalidPlan as error:
        return dict(valid=False,error=str(error))


def encodings(plan):
    desc={v['id']:v for v in plan['values']}
    units,ordinary=[],[]
    for step in steps(plan):
        if step['kind']!='encode': continue
        payload=step['payload']['values']
        scale=desc[step['output']]['scale_log2']
        row=dict(value=step['output'],scale_log2=scale)
        if len(payload)==8 and (all(v==1 for v in payload) or all(v==-1 for v in payload)):
            row['abs_error']=abs(decoded_unit(scale)-1)
            units.append(row)
        elif scale<40-1e-8:
            ordinary.append(row)
    return dict(unit_count=len(units),unit_below_legacy_floor=[r for r in units if r['scale_log2']<30-1e-8],
                unit_over_error_budget=[r for r in units if r['abs_error']>1e-6],
                max_local_unit_error=max((r['abs_error'] for r in units),default=0),
                ordinary_below_floor=ordinary)


def budget(before,after):
    mapping={}
    def bind(a,b):
        if a in mapping and mapping[a]!=b:
            raise InvalidPlan('inconsistent boundary correspondence')
        if a not in mapping and b in mapping.values():
            raise InvalidPlan('non-bijective boundary correspondence')
        mapping[a]=b
    for field in ('external_inputs','final_outputs'):
        if len(before[field])!=len(after[field]): raise InvalidPlan('changed boundary arity')
        for a,b in zip(before[field],after[field]): bind(a,b)
    boots=[[s for s in steps(plan) if s.get('op')=='boot'] for plan in (before,after)]
    if len(boots[0])!=len(boots[1]): raise InvalidPlan('changed Boot count')
    for a,b in zip(*boots):
        bind(a['inputs'][0],b['inputs'][0]);bind(a['output'],b['output'])
    contract=dict(format_version=1,boundaries=[dict(name=f'boundary-{i}',before=a,after=b)
        for i,(a,b) in enumerate(mapping.items())])
    return dict(valid=True,contract=contract,stats=verify_alignment_budget(before,after,contract),
                scope='inputs, final outputs, and ordered Boot endpoints; no internal semantic-ID mapping')


def write_report(output,summary):
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    rows=[]
    def outcome(row):
        label=row['status']
        if row['status']=='valid': label+=f' / Boot={row["precision"]["stats"]["boot_count"]}'
        elif 'diagnostics' in row: label+=f' / raw Boot={row["diagnostics"]["boot_count"]}'
        return html.escape(label)
    for case in summary['cases']:
        for pair in case['runs']:
            links=[]
            for mode in ('off','on'):
                item=pair[mode]
                links.append(f'<td><a href="{case["case"]}/{pair["policy"]}-{pair["strategy"]}-{mode}.log">'
                             f'{outcome(item)}</a></td>')
            rows.append(f'<tr><td>{case["case"]}</td><td>{case["reference_boots"]}</td>'
                        f'<td>{pair["policy"]}/{pair["strategy"]}</td>'+''.join(links)+
                        f'<td>{html.escape(pair["result"])}</td></tr>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>大图 scale 对齐回归</title>'
        '<style>body{font:15px system-ui;margin:30px}table{border-collapse:collapse}td,th{border-bottom:1px solid #ddd;padding:9px}</style>'
        '<h1>ResNet20 / Qwen 模式大图回归</h1><p>同一编译器、相同输入、不同开关。默认判定使用单位量化契约：'
        '单位局部误差 ≤ 1e-6，最终输出使用原案例的绝对/相对容差；旧的编码 scale 下限结果也保存在 JSON 中。'
        'raw Boot 仅是非法输出的诊断，不能用于算法优劣比较。未执行真实加密。</p>'
        '<a href="summary.json">完整统计、命令及错误详情</a><table><tr><th>Case</th><th>参考 Boot</th>'
        '<th>策略</th><th>关闭 pass</th><th>开启 pass</th><th>配对结果</th></tr>'+''.join(rows)+'</table>')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hecate-opt',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--suite',type=Path,default=ROOT/'cases/model-large-q50')
    parser.add_argument('--family',choices=('resnet20','qwen','all'),default='all')
    parser.add_argument('--policies',nargs='+',choices=('threshold','regional'),default=['threshold','regional'])
    parser.add_argument('--strategies',nargs='+',choices=('greedy','depth-dp','optimized'),default=['greedy','optimized'])
    parser.add_argument('--timeout',type=float,default=120)
    parser.add_argument('--regional-output-scale',type=float,
                        help='Explicit compiler output target for regional runs; fixture limits stay unchanged')
    args=parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout<=0: parser.error('timeout must be finite and positive')
    if args.regional_output_scale is not None and (not math.isfinite(args.regional_output_scale) or args.regional_output_scale<=0):
        parser.error('regional output scale must be finite and positive')
    suite,output,compiler=args.suite.resolve(),args.output_dir.resolve(),args.hecate_opt.resolve()
    output.mkdir(parents=True,exist_ok=True)
    spec_path=suite/'operator-spec.json';spec=read_json(spec_path)
    digest='sha256:'+hashlib.sha256(spec_path.read_bytes()).hexdigest()
    summary=dict(family=args.family,compiler=dict(path=str(compiler),sha256=hashlib.sha256(compiler.read_bytes()).hexdigest()),
                 operator_spec_sha256=digest,timeout_seconds=args.timeout,
                 policies=args.policies,strategies=args.strategies,regional_output_scale=args.regional_output_scale,
                 encrypted_accuracy_checked=False,cases=[])
    names=[n for n in read_json(suite/'suite.json')['cases'] if args.family=='all' or n.startswith(args.family+'-')]
    for name in names:
        directory=suite/name;folder=output/name;folder.mkdir(exist_ok=True)
        graph,case,reference=[read_json(directory/(stem+'.json')) for stem in ('graph','case','reference')]
        manual=reference_plan(directory)
        for key in ('graph','case','operator_spec','recipe','pattern'):
            path=spec_path if key=='operator_spec' else directory/(key+'.json')
            if reference['inputs_sha256'][key]!=hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError(f'{name}: frozen {key} digest mismatch')
        precision=deepcopy(case)
        precision['unit_encoding']=dict(max_abs_error=1e-6,output_tolerance=deepcopy(case['numeric_tolerance']))
        ref_legacy=check(manual,spec,case,digest);ref_precision=check(manual,spec,precision,digest)
        if not ref_legacy['valid'] or not ref_precision['valid']:
            raise ValueError(f'{name}: reference invalid: {ref_legacy}, {ref_precision}')
        (folder/'input.mlir').write_text(helpers['earth_source'](graph,case,spec))
        for stem,data in (('case',case),('precision-case',precision)):
            (folder/(stem+'.json')).write_text(json.dumps(data,indent=2)+'\n')
        row=dict(case=name,input_compute_ops=len(graph['ops']),reference_boots=ref_precision['stats']['boot_count'],
                 reference_legacy=ref_legacy,reference_precision=ref_precision,
                 inputs_sha256=reference['inputs_sha256'],runs=[])
        summary['cases'].append(row)
        for policy in args.policies:
            for strategy in args.strategies:
                pair=dict(policy=policy,strategy=strategy)
                plans={}
                for mode in ('off','on'):
                    stem=f'{policy}-{strategy}-{mode}';prefix=folder/stem
                    cmd=[str(compiler),str(folder/'input.mlir'),'--dacapo',f'--boot-placement={strategy}',
                        f'--native-scale-policy={policy}',f'--ckks-config={suite}/compiler.json','--waterline=40','--bootstrap-scale=40',
                        '--polynomial-optimizations=none','--mlir-print-op-on-diagnostic=false',
                        '--runtime-plan-format=json','--runtime-plan-inline-payload-max-bytes=1048576',
                        '--runtime-plan-levels-per-logical-level=1',
                        f'--runtime-plan-physical-level-operator-spec-path={spec_path}',f'--runtime-plan-operator-spec-path={spec_path}',
                        f'--runtime-plan-boot-profile={spec["boot_profiles"][0]["profile_id"]}',
                        f'--runtime-plan-target-id={spec["target_id"]}',f'--runtime-plan-context-id={spec["context"]["context_id"]}',
                        f'--runtime-plan-operator-spec-id={spec["spec_id"]}',f'--runtime-plan-operator-spec-version={spec["version"]}',
                        f'--runtime-plan-operator-spec-sha256={digest}','-o',str(prefix)+'.mlir']
                    if mode=='on': cmd.append('--enable-scale-alignment-hoisting')
                    if policy=='regional' and args.regional_output_scale is not None:
                        cmd.append(f'--regional-output-scale={args.regional_output_scale}')
                    (folder/(stem+'.command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
                    plan_path=Path(str(prefix)+'.graph.runtime-plan.json');plan_path.unlink(missing_ok=True)
                    start=time.perf_counter()
                    try:
                        run=subprocess.run(cmd,capture_output=True,text=True,timeout=args.timeout)
                    except subprocess.TimeoutExpired as error:
                        (folder/(stem+'.log')).write_bytes((error.stdout or b'')+(error.stderr or b''))
                        result=dict(status='timeout')
                    else:
                        (folder/(stem+'.log')).write_text(run.stdout+run.stderr)
                        if run.returncode:
                            result=dict(status='compiler_crashed' if run.returncode<0 else 'compile_failed',
                                        returncode=run.returncode,error=run.stderr,
                                        stage='alignment_pass' if 'hoist-scale-alignment' in run.stderr else 'compiler_before_or_after_alignment')
                        else:
                            plan=read_json(plan_path);plans[mode]=plan
                            legacy=check(plan,spec,case,digest);quantized=check(plan,spec,precision,digest)
                            result=dict(status='valid' if quantized['valid'] else 'invalid',legacy=legacy,precision=quantized,
                                        diagnostics=helpers['plan_diagnostics'](plan),encodings=encodings(plan),
                                        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest())
                            if quantized['valid']:
                                result['reference_boot_delta']=quantized['stats']['boot_count']-row['reference_boots']
                    result['compile_seconds']=time.perf_counter()-start
                    pair[mode]=result
                    print(name,policy,strategy,mode,result['status'],f'{result["compile_seconds"]:.3f}s',flush=True)
                if all(m in plans for m in ('off','on')):
                    try: pair['boundary_check']=budget(plans['off'],plans['on'])
                    except InvalidPlan as error: pair['boundary_check']=dict(valid=False,error=str(error))
                a,b=pair['off']['status'],pair['on']['status']
                pair['result']=('both_valid' if a==b=='valid' else 'regression' if a=='valid' else
                                'repaired' if b=='valid' else 'neither_valid')
                if a==b=='valid':
                    pair['boot_delta']=pair['on']['precision']['stats']['boot_count']-pair['off']['precision']['stats']['boot_count']
                row['runs'].append(pair)
                summary['outcomes']=dict(Counter(p[m]['status'] for r in summary['cases'] for p in r['runs'] for m in ('off','on')))
                write_report(output,summary)
    print('report:',output/'index.html',flush=True)
    return int(any(p[m]['status']!='valid' for r in summary['cases'] for p in r['runs'] for m in ('off','on')))


if __name__=='__main__':
    raise SystemExit(main())
