#!/usr/bin/env python3
"""Check rolling Prepare lifetime, placement/order, and complete-group capacity."""
import argparse
from collections import Counter
import json
import subprocess
import tempfile
from pathlib import Path


def source(count):
    c = 'tensor<1x!ckks.poly<2 * 40 * 1>>'
    p = 'tensor<1x!ckks.poly<1 * 40 * 1>>'
    rows = ['module {', f'func.func @waterline(%arg0: {c} {{dist.rank = 0 : i64, dist.device = 0 : i64}}) -> {c} attributes {{dist.device_counts = array<i64: 1>}} {{',
        f'%w = "ckks.encode"() <{{payload = dense<2.0> : tensor<1xf64>}}> {{dist.logical_id = 1 : i64, dist.rank = 0 : i64, dist.device = -1 : i64}} : () -> {p}',
        f'%p = "dist.transfer"(%w) <{{transfer_id = 0 : i64, source_rank = 0 : i64, source_device = -1 : i64, destination_rank = 0 : i64, destination_device = 0 : i64, initialization = true}}> {{dist.rank = 0 : i64, dist.device = 0 : i64}} : ({p}) -> {p}']
    for i in range(count):
        arg = f'%v{i-1}' if i else '%arg0'
        rows.append(f'%v{i} = "ckks.addcp"({arg}, %p) {{dist.logical_id = {i+2} : i64, dist.rank = 0 : i64, dist.device = 0 : i64}} : ({c}, {p}) -> {c}')
    rows += [f'return %v{count-1} : {c}', '}', '}']
    return '\n'.join(rows)


def heft_fanout_source():
    """Unplaced fanout with two shared weights and a Host refresh boundary."""
    ct = 'tensor<1x!ckks.poly<2 * 40 * 5>>'
    pt = 'tensor<1x!ckks.poly<1 * 40 * 5>>'
    rows = ['module {', f'func.func @heft_waterline(%arg0: {ct}) -> {ct} {{']
    for index, scalar in enumerate((1.0, 2.0)):
        rows.append(f'%w{index} = "ckks.encode"() <{{payload = dense<{scalar}> : tensor<1xf64>}}> : () -> {pt}')
    offsets = (1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, 1, 2)
    leaves = []
    for index, offset in enumerate(offsets):
        rows.append(f'%r{index} = "ckks.rotatec"(%arg0) <{{offset = array<i64: {offset}>}}> : ({ct}) -> {ct}')
        rows.append(f'%a{index} = "ckks.addcp"(%r{index}, %w0) : ({ct}, {pt}) -> {ct}')
        rows.append(f'%b{index} = "ckks.addcp"(%a{index}, %w1) : ({ct}, {pt}) -> {ct}')
        leaves.append(f'%b{index}')
    reduction = 0
    while len(leaves) > 1:
        next_leaves = []
        for left, right in zip(leaves[::2], leaves[1::2]):
            name = f'%sum{reduction}'
            rows.append(f'{name} = "ckks.addcc"({left}, {right}) : ({ct}, {ct}) -> {ct}')
            next_leaves.append(name)
            reduction += 1
        leaves = next_leaves
    rows += [f'%boot = "ckks.bootstrapc"({leaves[0]}) : ({ct}) -> {ct}',
             f'%result = "ckks.negatec"(%boot) : ({ct}) -> {ct}',
             f'return %result : {ct}', '}', '}']
    return '\n'.join(rows)


def check_heft_preservation(args, temp, spec):
    # Run HEFT and communication materialization exactly once. Both exports
    # below consume that same placed IR; Waterline must not assign new places
    # or alter ciphertext computation/communication order during late rewrite.
    unplaced = temp / 'heft-unplaced.mlir'
    unplaced.write_text(heft_fanout_source())
    placed = temp / 'heft-placed.mlir'
    fixtures = args.source_dir / 'test/runtime-plan'
    placement = (f'assign-ckks-placement{{device-counts=4 operator-spec={spec} '
                 f'boot-profile=test-boot communication-profile={fixtures / "placement-communication-profile.json"}}}')

    def compile_ir(path, passes, output):
        return subprocess.run(
            [str(args.hecate_opt), str(path),
             f'-p=builtin.module(func.func({",".join(passes)}))', '-o', str(output)],
            capture_output=True, text=True, timeout=90, cwd=args.source_dir)

    assigned = compile_ir(unplaced, [placement, 'materialize-ckks-communication'], placed)
    assert assigned.returncode == 0, assigned.stderr
    placed_text = placed.read_text()
    assert 'dist.schedule_start' in placed_text
    plans = {}
    for mode in ('eager', 'waterline'):
        prefix = temp / f'heft-{mode}'
        passes = []
        if mode == 'waterline':
            passes.append(f'plan-plaintext-waterline{{operator-spec={spec} prefix={prefix} '
                          'gpu-budget-bytes=1048576 fraction=0.5 host-budget-bytes=1048576 '
                          'pinned-budget-bytes=1048576 raw-budget-bytes=128}')
        passes.append(f'emit-runtime-plan{{plan-format=json prefix={prefix} target-id=test '
                      f'operator-spec-id=test operator-spec-sha256=sha256:{"0"*64} '
                      'context-id=test-context boot-profile=test-boot boot-implementation=decrypt_reencrypt}')
        result = compile_ir(placed, passes, temp / f'heft-{mode}.mlir')
        assert result.returncode == 0, result.stderr
        plans[mode] = json.loads(Path(f'{prefix}.heft_waterline.runtime-plan.json').read_text())
    assert placed.read_text() == placed_text

    def foreground(plan):
        ciphertexts = {value['id'] for value in plan['values'] if value['kind'] == 'ciphertext'}
        operations = []
        for phase in ('initialization', 'execution'):
            for instruction in plan[phase]:
                is_compute = instruction['kind'] == 'compute' and instruction['output'] in ciphertexts
                is_transfer = instruction['kind'] == 'transfer' and any(
                    output in ciphertexts for output in instruction['outputs'])
                if is_compute or is_transfer:
                    # Removing Host Encode changes absolute plan ordinals.
                    # Compare the foreground subsequence and every semantic
                    # field, retaining the original phase and relative order.
                    operations.append((phase, {key: value for key, value in instruction.items()
                                               if key != 'ordinal'}))
        return operations

    eager, waterline = plans['eager'], plans['waterline']
    expected = foreground(eager)
    actual = foreground(waterline)
    # Instruction equality checks places, operand/result IDs, operator
    # parameters, CT transfer IDs/endpoints and relative order in each phase.
    assert actual == expected, 'Waterline changed a placed foreground instruction or its order'
    for field in ('external_inputs', 'final_outputs'):
        assert waterline[field] == eager[field], field
    def ciphertext_values(plan):
        return {value['id']: value for value in plan['values'] if value['kind'] == 'ciphertext'}
    assert ciphertext_values(waterline) == ciphertext_values(eager)

    computes = [instruction for _, instruction in actual if instruction['kind'] == 'compute']
    device_places = {instruction['place']['index'] for instruction in computes
                     if instruction['place']['kind'] == 'device'}
    assert len(device_places) >= 2, 'HEFT fixture did not exercise multiple GPU placements'
    assert len(computes) == 65
    boots = [instruction for instruction in computes if instruction['op'] == 'boot']
    assert len(boots) == 1 and boots[0]['place']['kind'] == 'host'
    transfers = [instruction for _, instruction in actual if instruction['kind'] == 'transfer']
    assert any(instruction['sources'][0]['kind'] == 'device' and
               instruction['destinations'][0]['kind'] == 'device' and
               instruction['sources'][0] != instruction['destinations'][0]
               for instruction in transfers), 'HEFT fixture has no inter-GPU ciphertext edge'
    assert any(instruction['sources'][0]['kind'] == 'device' and
               instruction['destinations'][0]['kind'] == 'host' for instruction in transfers)
    assert any(instruction['sources'][0]['kind'] == 'host' and
               instruction['destinations'][0]['kind'] == 'device' for instruction in transfers)
    instructions = waterline['initialization'] + waterline['execution']
    prepared = {instruction['output'] for instruction in instructions
                if instruction['kind'] == 'prepare_plaintext'}
    uses = Counter(operand for instruction in computes for operand in instruction['inputs'] if operand in prepared)
    assert len(prepared) >= 4 and all(uses[value] >= 2 for value in prepared)
    assert not any(instruction['kind'] in ('encode', 'fence') for instruction in instructions)
    assert not any(instruction['kind'] == 'transfer' and instruction['output_kinds'] == ['plaintext']
                   for instruction in instructions)
    print(f'PASS: actual HEFT on {len(device_places)} GPUs, same placed IR eager/Waterline: '
          f'{len(computes)} CT computes, {len(transfers)} CT transfers, Host Boot, '
          f'{len(prepared)} shared Prepare outputs; places/parameters/IDs/order identical')


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
        plan=json.loads(Path(f'{prefix}.waterline.runtime-plan.json').read_text())
        assert plan['format_version']==4 and 'waterline' in plan
        all_ops=plan['initialization']+plan['execution']
        assert sum(i['kind']=='prepare_plaintext' for i in all_ops)==1
        assert not any(i['kind'] in ('fence','encode','transfer') for i in all_ops)
        compute=[i for i in all_ops if i['kind']=='compute']
        assert len(compute)==200 and all(i['place']=={'kind':'device','rank':0,'index':0} for i in compute)
        assert [i['output'] for i in compute]==[str(i+2) for i in range(200)]
        report=json.loads(Path(f'{prefix}.waterline.waterline.json').read_text())
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
            parsed=json.loads(Path(f'{prefix}.waterline.runtime-plan.json').read_text())
            instructions=parsed['initialization']+parsed['execution']
            assert sum(i['kind']=='prepare_plaintext' for i in instructions)==1
            assert sum(i['kind']=='compute' for i in instructions)==200
            assert not any(i['kind']=='fence' for i in instructions)
        # GPU's raw-double/padded-pinned buffers already fit the raw Host
        # allowance. Fixing CPU must not add its RNS upload staging here.
        gpu=run(4096,extra='backend=gpu',host_budget=64,operator_spec=small_spec_path)
        assert gpu.returncode==0,gpu.stderr
        print('PASS: degree8 CPU/Auto Host480 rejects, Host512 completes unchanged 200-consumer graph; GPU Host64 remains feasible')
        check_heft_preservation(args, temp, spec)

if __name__=='__main__': main()
