#!/usr/bin/env python3
"""Check rolling Prepare lifetime, placement/order, and complete-group capacity."""
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
        def run(budget, fraction=.5, extra='', host_budget=16384, operator_spec=spec):
            passes=[f'plan-plaintext-waterline{{operator-spec={operator_spec} prefix={prefix} gpu-budget-bytes={budget} fraction={fraction} host-budget-bytes={host_budget} pinned-budget-bytes=8192 raw-budget-bytes=128 {extra}}}',
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
        # A CPU-encoded Host plaintext stays alive while its separate pinned
        # uint32 upload buffer is in flight. This real compiler graph has one
        # shared plaintext and 200 consumers, so only the first group needs
        # preparation space. The 480-byte budget previously admitted it while
        # omitting pinned storage from Host; 512 bytes covers the complete group.
        small_spec=json.loads(spec.read_text());small_spec['context']['poly_degree']=8
        small_spec_path=temp/'degree8-spec.json';small_spec_path.write_text(json.dumps(small_spec))
        for backend in ('cpu','auto'):
            bad=run(4096,extra=f'backend={backend}',host_budget=480,operator_spec=small_spec_path)
            assert bad.returncode!=0 and 'single-group shared Host/pinned budget exceeded' in bad.stderr, \
                f'{backend} omitted CPU upload pinned bytes from Host at 480: exit={bad.returncode} {bad.stderr}'
            good=run(4096,extra=f'backend={backend}',host_budget=512,operator_spec=small_spec_path)
            assert good.returncode==0,good.stderr
            parsed=json.loads(Path(f'{prefix}.stream.runtime-plan.json').read_text())
            instructions=parsed['initialization']+parsed['execution']
            assert sum(i['kind']=='prepare_plaintext' for i in instructions)==1
            assert sum(i['kind']=='compute' for i in instructions)==200
            assert not any(i['kind']=='fence' for i in instructions)
        # GPU's raw-double/padded-pinned buffers already fit the raw Host
        # allowance. Fixing CPU must not add its RNS upload staging here.
        gpu=run(4096,extra='backend=gpu',host_budget=64,operator_spec=small_spec_path)
        assert gpu.returncode==0,gpu.stderr
        print('PASS: degree8 CPU/Auto Host480 rejects, Host512 completes unchanged 200-consumer graph; GPU Host64 remains feasible')

if __name__=='__main__': main()
