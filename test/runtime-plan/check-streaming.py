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


def simulate(plan, budget):
    values = {v['id']: v for v in plan['values']}
    live = {plan['external_inputs'][0]: 1.0}
    allocated = {plan['external_inputs'][0]}
    peak = 0
    transfers = set()
    for op in plan['initialization'] + plan['execution']:
        kind = op['kind']
        if kind == 'fence':
            allocated = set(live) | set(plan["external_inputs"])
        elif kind == 'release':
            del live[op['value']]
        elif kind == 'encode':
            assert op['output'] not in live
            live[op['output']] = op['payload']['values'][0]
            allocated.add(op['output'])
        elif kind == 'transfer':
            assert op['transfer_id'] not in transfers
            transfers.add(op['transfer_id'])
            live[op['outputs'][0]] = live[op['inputs'][0]]
            allocated.add(op['outputs'][0])
        else:
            live[op['output']] = sum(live[x] for x in op['inputs'])
            allocated.add(op['output'])
        gpu = sum(16 * (values[v]['level'] + 1) * values[v]['components'] * 4
                  for v in allocated if values[v]['place']['kind'] == 'device')
        assert gpu <= budget, (gpu, budget)
        peak = max(peak, gpu)
    return live[plan['final_outputs'][0]], peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hecate-opt', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as d:
        temp = Path(d)
        spec = args.source_dir / 'test/runtime-plan/placement-operator-spec.json'
        def compile(count, gpu=1408, prefetch=0, raw=128, host=4096, pinned=4096):
            path = temp / 'input.mlir'
            path.write_text(source(count))
            prefix = temp / 'out'
            passes = [f'plan-plaintext-streaming{{operator-spec={spec} prefix={prefix} gpu-budget-bytes={gpu} host-budget-bytes={host} pinned-budget-bytes={pinned} raw-budget-bytes={raw} prefetch-bytes={prefetch}}}',
                      'plan-runtime-memory{enable-reuse=false}',
                      f'emit-runtime-plan{{prefix={prefix} target-id=test operator-spec-id=test operator-spec-sha256=sha256:{"0"*64} context-id=test-context}}']
            result = subprocess.run([str(args.hecate_opt), str(path), f'-p=builtin.module(func.func({",".join(passes)}))', '-o', str(temp/'out.mlir')], capture_output=True, text=True, timeout=90, cwd=args.source_dir)
            if result.returncode:
                return result.stderr
            plan = json.loads(Path(f'{prefix}.stream.runtime-plan.json').read_text())
            report = json.loads(Path(f'{prefix}.stream.streaming.json').read_text())
            assert plan['format_version'] == 3 and not plan['initialization']
            answer, peak = simulate(plan, gpu)
            assert answer == 1 + count*2
            assert peak == report['object_peak_bytes_host_then_gpu'][1]
            assert report['pinned_peak_bytes'] <= pinned and report['raw_peak_bytes'] <= raw
            assert report['encodes'] > 1
            return report
        for prefetch in [0, 4096]:
            report = compile(12, prefetch=prefetch)
            assert isinstance(report, dict), report
        for options, message in [({'gpu': 639}, 'GPU0 requires'), ({'host': 255}, 'Host requires'),
                                 ({'pinned': 127}, 'pinned requires'), ({'raw': 15}, 'raw Encode payload')]:
            error = compile(12, **options)
            assert isinstance(error, str) and message in error, error
        # A hand-authored Fence must survive CSE/DCE and select V3 without
        # relying on an attribute installed by the streaming pass.
        fence = temp / 'fence.mlir'
        fence.write_text(source(1).replace('return %v0', '"dist.fence"() : () -> ()\nreturn %v0'))
        subprocess.run([str(args.hecate_opt), str(fence),
            f'-p=builtin.module(func.func(canonicalize,cse,emit-runtime-plan{{prefix={temp}/fence target-id=test operator-spec-id=test operator-spec-sha256=sha256:{"0"*64} context-id=test-context}}))',
            '-o', str(temp/'fence.out.mlir')], cwd=args.source_dir, check=True, capture_output=True)
        emitted = json.loads((temp/'fence.stream.runtime-plan.json').read_text())
        assert emitted['format_version'] == 3
        assert sum(op['kind'] == 'fence' for op in emitted['execution']) == 1
        report = compile(20000)
        assert isinstance(report, dict), report
        print('PASS streaming equivalence, independent retained-allocation bounds, all budgets, re-encoding, prefetch, 20000 operations')

if __name__ == '__main__':
    main()
