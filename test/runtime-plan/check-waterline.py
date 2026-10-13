#!/usr/bin/env python3
"""Independent batch-lifetime simulation, equivalence, rejection and scale checks."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def source(count):
    c = 'tensor<1x!ckks.poly<2 * 40 * 1>>'
    p = 'tensor<1x!ckks.poly<1 * 40 * 1>>'
    rows = ['module {', f'func.func @stream(%arg0: {c} {{dist.rank = 0 : i64, dist.device = 0 : i64}}) -> {c} attributes {{dist.device_counts = array<i64: 1>}} {{',
        f'%w = "ckks.encode"() <{{payload = dense<2.0> : tensor<1xf64>}}> {{dist.logical_id = 1 : i64, dist.rank = 0 : i64, dist.device = -1 : i64}} : () -> {p}',
        f'%p = "dist.transfer"(%w) <{{transfer_id = 0 : i64, source_rank = 0 : i64, source_device = -1 : i64, destination_rank = 0 : i64, destination_device = 0 : i64, initialization = true}}> {{dist.rank = 0 : i64, dist.device = 0 : i64}} : ({p}) -> {p}']
    for i in range(count):
        arg = f'%v{i-1}' if i else '%arg0'
        rows.append(f'%v{i} = "ckks.addcp"({arg}, %p) {{dist.logical_id = {i+2} : i64, dist.rank = 0 : i64, dist.device = 0 : i64}} : ({c}, {p}) -> {c}')
    rows += [f'return %v{count-1} : {c}', '}', '}']
    return '\n'.join(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hecate-opt', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    args=parser.parse_args()
    args.hecate_opt=args.hecate_opt.resolve();args.source_dir=args.source_dir.resolve()
    with tempfile.TemporaryDirectory() as d:
        temp=Path(d);path=temp/'input.mlir';path.write_text(source(200))
        spec=args.source_dir/'test/runtime-plan/placement-operator-spec.json'
        prefix=temp/'out'
        def run(budget, fraction=.5, extra=''):
            passes=[f'plan-plaintext-waterline{{operator-spec={spec} prefix={prefix} gpu-budget-bytes={budget} fraction={fraction} host-budget-bytes=16384 pinned-budget-bytes=8192 raw-budget-bytes=128 {extra}}}',
                'plan-runtime-memory{enable-reuse=false}',
                f'emit-runtime-plan{{plan-format=json prefix={prefix} target-id=test operator-spec-id=test operator-spec-sha256=sha256:{"0"*64} context-id=test-context}}']
            return subprocess.run([str(args.hecate_opt),str(path),f'-p=builtin.module(func.func({",".join(passes)}))','-o',str(temp/'out.mlir')],capture_output=True,text=True,timeout=90,cwd=args.source_dir)
        result=run(4096);assert result.returncode==0,result.stderr
        plan=json.loads(Path(f'{prefix}.stream.runtime-plan.json').read_text())
        assert plan['format_version']==4 and 'waterline' in plan
        all_ops=plan['initialization']+plan['execution']
        assert sum(i['kind']=='prepare_plaintext' for i in all_ops)==1
        assert not any(i['kind'] in ('fence','encode','transfer') for i in all_ops)
        compute=[i for i in all_ops if i['kind']=='compute']
        assert len(compute)==200 and all(i['place']=={'kind':'device','rank':0,'index':0} for i in compute)
        assert [i['output'] for i in compute]==[str(i+2) for i in range(200)]
        report=json.loads(Path(f'{prefix}.stream.waterline.json').read_text())
        assert report['prepare_count']==1 and report['single_group_object_and_workspace_peak'][1]<4096
        for budget,fraction,extra in [(300,.5,''),(4096,.01,''),(4096,.5,'workspace-bytes-per-op=8192')]:
            bad=run(budget,fraction,extra);assert bad.returncode!=0 and 'single-group' in bad.stderr,bad.stderr
        print('PASS: Prepare V4, no automatic Fence/reencode, frontend places/order, single-group rejections')

if __name__=='__main__': main()
