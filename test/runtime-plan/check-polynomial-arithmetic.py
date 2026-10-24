#!/usr/bin/env python3
"""Check scalar rewrites, masks, shared results and managed scale semantics."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import runpy
import subprocess
import tempfile

CT = 'tensor<8x!earth.ci<0 * 0>>'
PT = 'tensor<8x!earth.pl<0 * 0>>'


def constant(name, value, length=8):
    return (f'%{name} = "earth.constant"() {{value = dense<{value}> : tensor<{length}xf64>, '
            f'rms_var = 1.0 : f64}} : () -> {PT}')


def mul(name, left, right, plain=False, swap=False):
    lt, rt = CT, PT if plain else CT
    if swap: left, right, lt, rt = right, left, rt, lt
    return f'%{name} = "earth.mul"(%{left}, %{right}) : ({lt}, {rt}) -> {CT}'


def source(lines, outputs):
    types = ', '.join([CT]*len(outputs))
    return (f'module {{ func.func @polynomial(%x: {CT}) -> ({types}) {{\n' +
            '\n'.join(lines) + '\nreturn ' + ', '.join('%'+v for v in outputs) +
            f' : {types}\n}}}}\n')


def replay(ir):
    # This checks scaled coefficient semantics with exact 2^30 drops and ideal
    # Boot. It does not simulate encrypted rounding/noise or claim GPU accuracy.
    values = {'%arg0': [v*2**40 for v in [-.4,-.3,-.2,-.1,.1,.2,.3,.4]]}
    types = {'%arg0': (2, 40, 19)}
    kinds, payloads = {}, {}
    counts = dict(boots=0, rescale_q=0, unit_muls=0, double_muls=0)
    for line in ir.splitlines():
        m = re.search(r'(%\w+) = "ckks\.(\w+)"\(([^)]*)\)', line)
        if not m: continue
        name, kind, inputs = m.groups()
        inputs = re.findall(r'%\w+', inputs)
        ts = [tuple(map(int, t)) for t in re.findall(r'!ckks.poly<(\d+) \* (\d+) \* (\d+)>', line)]
        ty = ts[-1]
        if kind == 'encode':
            payload = re.search(r'payload = dense<([^>]+)>', line)[1]
            if payload.startswith('['):
                array = [float(v) for v in payload.strip('[]').split(',')]
            else:
                length = int(re.search(r'dense<[^>]+> : tensor<(\d+)xf64>', line)[1])
                array = [float(payload)]*length
            payloads[name] = array
            result = [v*2**ty[1] for v in array+[0]*(8-len(array))]
        else:
            data = [values[v] for v in inputs]
            before = [types[v] for v in inputs]
            if kind in ('mulcp','mulcc'):
                assert before[0][2]==before[1][2]==ty[2]
                assert ty[1]==before[0][1]+before[1][1]
                result = [a*b for a,b in zip(*data)]
                if kind=='mulcp':
                    p=payloads.get(inputs[1],[])
                    counts['unit_muls'] += p==[1]*8
                    counts['double_muls'] += p==[2]*8
            elif kind in ('addcc','addcp'):
                assert before[0][1:]==before[1][1:]==ty[1:], line
                result = [a+b for a,b in zip(*data)]
            elif kind=='rescalec':
                assert before[0][1]>=70 and ty[1]==before[0][1]-30
                assert ty[2]==before[0][2]-1
                result = [v/2**30 for v in data[0]]
                counts['rescale_q']+=1
            elif kind=='bootstrapc':
                assert ty[1:]==(45,19)
                result=[v*2**(45-before[0][1]) for v in data[0]]
                counts['boots']+=1
            elif kind in ('modswitchc','relinearize'):
                assert ty[1]==before[0][1]
                result=data[0]
            elif kind=='negatec': result=[-v for v in data[0]]
            else: raise AssertionError(kind)
        values[name]=result; types[name]=ty; kinds[name]=kind
    ret=next(l for l in ir.splitlines() if l.strip().startswith('return '))
    outputs=re.findall(r'%\w+',ret.split(':')[0])
    return [[v/2**types[o][1] for v in values[o]] for o in outputs], counts


def main():
    p=argparse.ArgumentParser();p.add_argument('--hecate-opt',type=Path,required=True)
    p.add_argument('--source-dir',type=Path,required=True);a=p.parse_args()
    with tempfile.TemporaryDirectory(prefix='polynomial-arithmetic-') as directory:
        root=Path(directory)
        config=json.loads((a.source_dir/'config.json').read_text())
        config.update(polynomialDegree=16,rescalingFactor=30,levelLowerBound=1,
                      levelUpperBound=19,bootstrapLevelLowerBound=1,bootstrapLevelUpperBound=19)
        profile=root/'profile.json';profile.write_text(json.dumps(config))
        def compile(text,mode,strategy='greedy',pipeline=None):
            path=root/'input.mlir';out=root/'output.mlir';path.write_text(text)
            command=[str(a.hecate_opt),f'--ckks-config={profile}']
            if mode is not None:
                command.append(f'--polynomial-optimizations={mode}')
            command += ([f'--pass-pipeline={pipeline}'] if pipeline else
                        ['--dacapo-precompile',f'--boot-placement={strategy}',
                         '--waterline=40','--bootstrap-scale=45'])
            r=subprocess.run([*command,str(path),'-o',str(out)],text=True,capture_output=True)
            assert r.returncode==0,r.stderr
            return out.read_text()
        cases=[]
        for coefficient,length,swap in [('2.0',8,False),('2.0',8,True),
                                         ('2.0',3,False),('2.000000000000001',8,False),
                                         ('[2.,2.,0.,2.,2.,2.,2.,2.]',8,False)]:
            lines=[constant('p',coefficient,length),mul('d','x','p',True,swap),mul('out','d','x')]
            cases.append((source(lines,['out']),coefficient=='2.0' and length==8))
        # Different source scales/levels force alignment after a real Rescale.
        for shared in (False,True):
            lines=[constant('p','0.125'),constant('q','0.3'),mul('square','x','x'),
                   mul('a','x','p',True),mul('b','square','q',True),
                   f'%sum = "earth.add"(%a,%b) : ({CT},{CT}) -> {CT}']
            cases.append((source(lines,['sum','a'] if shared else ['sum']),False))
        lines=[constant('p','0.000000000931322574615478515625'),mul('out','x','p',True)]
        cases.append((source(lines,['out']),False))
        # Shared plaintext consumers must agree in segment simulation and final
        # emission after Boot changes the running scale from 40 to 45.
        lines=[constant('p','1.01')]
        previous='x'
        for i in range(24):
            name=f'chain{i}';lines.append(mul(name,previous,'p',True));previous=name
        cases.append((source(lines,[previous]),False))
        for text,doubling in cases:
            for strategy in ['greedy','depth-dp','optimized']:
                baseline,old=replay(compile(text,'none',strategy))
                for mode in ['double','double-align']:
                    actual,counts=replay(compile(text,mode,strategy))
                    assert len(actual)==len(baseline)
                    for got,want in zip(actual,baseline):
                        assert all(math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-12) for x,y in zip(got,want))
                    if doubling:
                        assert old['double_muls']==1 and counts['double_muls']==0
                        assert counts['rescale_q']<old['rescale_q']
        alignment=cases[5][0]
        _,before=replay(compile(alignment,'double'))
        _,after=replay(compile(alignment,'double-align'))
        assert after['unit_muls']<before['unit_muls']
        assert compile(alignment,None)==compile(alignment,'double-align')
        managed=source([constant('p','2.0'),mul('out','x','p',True)],['out'])
        managed=managed.replace('-> ('+CT+') {','-> ('+CT+') attributes {init_level = 19 : i64} {')
        ir=compile(managed,'double-align',pipeline='builtin.module(func.func(optimize-polynomial-arithmetic))')
        assert '"earth.mul"' in ir and '"earth.add"' not in ir
        # Removed experimental policies must fail instead of being accepted
        # silently through either the driver or the individual pass.
        for option in ['--polynomial-optimizations=all',
                       '--coefficient-min-scale=30',
                       '--coefficient-precision-bits=24',
                       '--pass-pipeline=builtin.module(func.func(optimize-polynomial-arithmetic{coefficient-min-scale=30}))',
                       '--pass-pipeline=builtin.module(func.func(optimize-polynomial-arithmetic{coefficient-precision-bits=24}))']:
            r=subprocess.run([str(a.hecate_opt),f'--ckks-config={profile}',
                              option,str(root/'input.mlir'),
                              '-o',str(root/'rejected.mlir')],text=True,capture_output=True)
            assert r.returncode!=0,(option,r.stdout,r.stderr)
            removed='all' if option.endswith('=all') else re.search(r'coefficient-[a-z-]+',option)[0]
            assert removed in r.stderr,(option,r.stdout,r.stderr)
        # Exercise real physical-scale propagation and RuntimePlan export too;
        # powers of two in the logical replay alone cannot catch prime drift.
        helpers=runpy.run_path(str(a.source_dir/'test/runtime-plan/check-threshold-graphs.py'))
        graph=helpers['Graph']()
        square=graph.op('mul','x','x')
        left=graph.op('mul','x','p');right=graph.op('mul',square,'p')
        graph.outputs=[graph.op('add',left,right),left]
        path=root/'physical.mlir'
        path.write_text(graph.source().replace('tensor<1xf64>','tensor<8xf64>'))
        spec=json.loads((a.source_dir/'test/runtime-plan/lazy-physical-level-operator-spec.json').read_text())
        for entry in spec['operators'].values():entry.update(supported=True,latency_us_by_level=[10]*46)
        spec['context']['default_scale_log2']=40
        spec['context']['rns_moduli']=[(1<<30)-(i+1)*32768+1 for i in range(50)]
        spec['rescale_mode']='native'
        spec['boot_profiles']=[dict(profile_id='polynomial',implementation='native',
            input_level_min=4,input_level_max=49,input_components=2,output_level=22,
            output_scale_log2=45,output_components=2,latency_us_by_input_level=[85000]*46,
            noise_by_input_level=None,host_requirements=dict(needs_secret_key=False,needs_host_compute=False))]
        operator=root/'spec.json';operator.write_text(json.dumps(spec))
        for mode in ['none','double','double-align']:
            for strategy in ['greedy','depth-dp','optimized']:
                out=root/f'physical-{mode}-{strategy}.mlir'
                cmd=[str(a.hecate_opt),'--dacapo',f'--boot-placement={strategy}',
                     f'--polynomial-optimizations={mode}',f'--ckks-config={profile}',
                     '--waterline=40','--bootstrap-scale=45','--runtime-plan-format=json',
                     '--runtime-plan-memory=true','--runtime-plan-memory-reuse=true',
                     '--runtime-plan-boot-profile=polynomial','--runtime-plan-boot-implementation=native',
                     f'--runtime-plan-physical-level-operator-spec-path={operator}',
                     '--runtime-plan-levels-per-logical-level=1','--runtime-plan-device-counts=1',
                     '--runtime-plan-device-count=1',f'--runtime-plan-operator-spec-path={operator}',
                     '--runtime-plan-target-id=lazy-physical-level-test',
                     '--runtime-plan-context-id=lazy-physical-level-test',
                     '--runtime-plan-operator-spec-id=lazy-physical-level-test-v1',
                     '--runtime-plan-operator-spec-version=1','--runtime-plan-operator-spec-sha256=sha256:'+
                     hashlib.sha256(operator.read_bytes()).hexdigest(),str(path),'-o',str(out)]
                r=subprocess.run(cmd,text=True,capture_output=True)
                assert r.returncode==0,r.stderr
                plan=json.loads(out.with_suffix('.graph.runtime-plan.json').read_text())
                helpers['check_plan'](plan,out.read_text(),graph,spec,0)
        print('Scalar arithmetic: 3 strategies x 3 modes, default mode, masks, shared values, managed IR and removed-option rejection passed')


if __name__=='__main__':main()
