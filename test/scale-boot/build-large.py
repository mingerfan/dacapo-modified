#!/usr/bin/env python3
"""Extract source arithmetic kernels and build reduced ResNet20/Qwen DAGs.

NumPy is needed only to expand the repository's ResNet Chebyshev tree.
Rebuilding reference plans from frozen graphs needs only the standard library.
"""
import argparse
import ast
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace

from reference import Infeasible, Plan, active_rows, materialize_recipe, reference_metadata
from verify import read_json, verify

ROOT = Path(__file__).resolve().parent
COMPILER = ROOT.parents[1]
POSEIDON = ROOT.parents[5]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def source_functions(path, names, namespace):
    """Execute only named local arithmetic functions, without module imports."""
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise ValueError(f"source functions changed: {path}")
    for n in selected:
        if n.decorator_list:
            raise ValueError("unexpected decorated arithmetic kernel")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return dict(path=str(path.relative_to(POSEIDON)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                functions={n.name: dict(line=n.lineno, end_line=n.end_lineno) for n in selected},
                extraction="execute original function AST with symbolic arithmetic")


class Expr:
    def __init__(self, graph, name):
        self.g, self.name = graph, name

    def __mul__(self, other):
        return self.g.binary('mul', self, other)

    __rmul__ = __mul__

    def __add__(self, other):
        return self.g.binary('add', self, other)

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other if isinstance(other, Expr) else -float(other))

    def __neg__(self):
        return self.g.op('negate', self)

    def rotate(self, steps):
        return self if steps % 8 == 0 else self.g.op('rotate', self, attrs=dict(steps=steps))


class Graph:
    def __init__(self, inputs=('x',)):
        self.data = dict(inputs=list(inputs), constants={}, ops=[], outputs=[])
        self.constant_ids = {}
        self.stages = []

    def input(self, name):
        if name not in self.data['inputs']:
            raise ValueError(name)
        return Expr(self, name)

    def constant(self, value):
        value = [float(x) for x in value] if isinstance(value, (list, tuple)) else float(value)
        key = json.dumps(value)
        if key not in self.constant_ids:
            name = f'p{len(self.constant_ids)}'
            self.constant_ids[key] = name
            self.data['constants'][name] = value
        return self.constant_ids[key]

    def binary(self, kind, a, b):
        name = b.name if isinstance(b, Expr) else self.constant(b)
        return self.op(kind, a, name)

    def op(self, kind, *operands, attrs=None):
        name = f'v{len(self.data["ops"])}'
        row = [name, kind, [v.name if isinstance(v, Expr) else v for v in operands]]
        if attrs:
            row.append(attrs)
        self.data['ops'].append(row)
        return Expr(self, name)

    def stage(self, label, function, *args):
        start = len(self.data['ops'])
        result = function(*args)
        self.stages.append(dict(name=label, first_op=f'v{start}', last_op=f'v{len(self.data["ops"])-1}'))
        return result

    def finish(self, result):
        self.data['outputs'] = [result.name]
        self.data['ops'] = active_rows(self.data)  # Drop source-local unused temporaries.
        used = {v for row in self.data['ops'] for v in row[2]}
        self.data['constants'] = {k:v for k,v in self.data['constants'].items() if k in used}
        return self


def total(x, size=8):
    if size != 8:
        x = x * [1. if i < size else 0. for i in range(8)]
    for shift in (1, 2, 4):
        x = x + x.rotate(shift)
    return x


def linear(x, seed=0, gain=.6):
    """8x8 diagonal BSGS, four shared baby rotations and two giant groups."""
    babies = [x.rotate(i) for i in range(4)]
    groups = []
    for giant in (0, 4):
        terms = []
        for baby in range(4):
            shift = giant + baby
            diagonal = [gain / 8 * (1 + .2 * math.sin((r+1)*(shift+1)+seed)) for r in range(8)]
            # Public roll(+giant) is undone by rotate(+giant) on the group.
            diagonal = diagonal[-giant:] + diagonal[:-giant] if giant else diagonal
            terms.append(babies[baby] * diagonal)
        group = terms[0]
        for term in terms[1:]:
            group = group + term
        groups.append(group.rotate(giant))
    return groups[0] + groups[1]


def conv_bn(x, seed=0):
    """Reduced one-pack 3x3 rotate/multiply/accumulate, fused BN gain/bias."""
    terms = []
    for tap, shift in enumerate((-3,-2,-1,-1,0,1,1,2,3)):
        weights = [.8/9 * (1 + .1 * math.cos(seed + tap + i)) for i in range(8)]
        terms.append(x.rotate(shift) * weights)
    value = terms[0]
    for term in terms[1:]:
        value = value + term
    return value + [.005 * (i-3.5) for i in range(8)]


def rope(x):
    angles = [1 / 10000 ** ((i % 4) / 4) for i in range(8)]
    a = x.rotate(4) * [-1. if i < 4 else 0. for i in range(8)]
    b = x.rotate(-4) * [0. if i < 4 else 1. for i in range(8)]
    return x * [math.cos(v) for v in angles] + (a+b) * [math.sin(v) for v in angles]


def definitions(nonlinear, resnet_silu):
    norm_profile = SimpleNamespace(domain=(.01, 4.), evaluation='newton', iterations=3)
    silu_profile = SimpleNamespace(squarings=1, exp_degree=6, intermediate_iterations=4, reciprocal_iterations=5)
    softmax_profile = SimpleNamespace(squarings=1, exp_degree=6, exp_bound=.5,
                                     intermediate_iterations=4, reciprocal_iterations=5)

    def norm(x):
        variance = total(x*x) * .125 + 1e-6
        inverse = nonlinear['inverse_sqrt'](variance, norm_profile, lambda c:c)
        return x * inverse * [.9 + .02*i for i in range(8)]

    def ffn(x):
        gate = linear(x, 1)
        gate = nonlinear['normalize_square_silu'](gate, silu_profile, lambda c:c)
        up = linear(x, 2)
        return linear(gate * up, 3)

    for count in (1, 2):
        g = Graph(); x = g.input('x')
        for block in range(count):
            residual = x
            x = g.stage(f'block{block}.conv1_bn', conv_bn, x, block*2)
            x = g.stage(f'block{block}.silu1', resnet_silu, x)
            x = g.stage(f'block{block}.conv2_bn', conv_bn, x, block*2+1)
            x = x + residual
            x = g.stage(f'block{block}.silu2', resnet_silu, x)
        yield f'resnet20-residual-{count}block', g.finish(x), dict(family='resnet20', blocks=count,
            pattern='ConvBN -> exact source SiLU kernel -> ConvBN -> residual add -> exact source SiLU kernel',
            reductions=['one ciphertext instead of all channels', '8 cyclic slots instead of full spatial layout',
                        'deterministic synthetic weights; original full SiLU tree/coefficients retained',
                        'one/two of the nine ResNet20 residual blocks; no full-model accuracy claim'])

    g = Graph(); x = g.input('x')
    for i in range(6):
        x = g.stage(f'norm{i}', norm, x)
        x = g.stage(f'projection{i}', linear, x, i, .8)
    yield 'qwen-rmsnorm-projections', g.finish(x), dict(family='qwen', pattern='six RMSNorm and BSGS projection stages',
        reductions=['8 hidden slots', '3 Newton rsqrt iterations; explicit domain [.01,4]', 'synthetic dense weights'])

    g = Graph(); x = g.input('x')
    for i in range(2):
        residual = x
        x = g.stage(f'block{i}.post_norm', norm, x)
        x = g.stage(f'block{i}.swiglu', ffn, x)
        x = x + residual
    yield 'qwen-swiglu-residual-2block', g.finish(x), dict(family='qwen', pattern='post RMSNorm -> gate/up -> SiLU(gate)*up -> down -> residual, twice',
        reductions=['8 hidden/FFN slots', 'synthetic dense weights', 'normalize-square SiLU: degree6, one square, reciprocal4/5',
                    'FFN portions only; attention not included'])

    g = Graph(inputs=('x','cached_k','cached_v')); residual = g.input('x')
    x = g.stage('pre_norm', norm, residual)
    q = g.stage('q_projection', linear, x, 4, 1.)
    k = g.stage('k_projection', linear, x, 5, 1.)
    v = g.stage('v_projection', linear, x, 6, 1.)
    q = g.stage('q_head_norm', norm, q)
    k = g.stage('k_head_norm', norm, k)
    q = g.stage('q_rope', rope, q); k = g.stage('k_rope', rope, k)
    current = total(q*k) * (1/math.sqrt(8))
    cached = total(q*g.input('cached_k')) * (1/math.sqrt(8))
    scores = current * [1.,0.,0.,0.,0.,0.,0.,0.] + cached * [0.,1.,0.,0.,0.,0.,0.,0.]
    centered = (scores + total(scores,2) * -.5) * [1.,1.,0.,0.,0.,0.,0.,0.]
    def softmax(z):
        return nonlinear['normalize_square'](z,softmax_profile,2,lambda c:c,
                                             lambda a:total(a,2),lambda a:a)
    probability = g.stage('normalize_square_softmax',softmax,centered)
    p0 = total(probability * [1.,0.,0.,0.,0.,0.,0.,0.])
    p1 = total(probability * [0.,1.,0.,0.,0.,0.,0.,0.])
    context = v*p0 + g.input('cached_v')*p1
    output = g.stage('o_projection',linear,context,7) + residual
    yield 'qwen-attention-cached-2tokens', g.finish(output), dict(family='qwen',
        pattern='pre RMSNorm -> Q/K/V -> Q/K head RMSNorm -> RoPE -> QK dot products -> normalize-square softmax -> weighted V -> O -> residual',
        reductions=['one query/head, one current and one cached visible key/value', '8 hidden/head slots',
                    'cached K is already normalized/RoPE-transformed', 'synthetic weights and KV values',
                    'degree6 exponential, one normalize-square round, reciprocal4/5; no FFN'])


def evaluate(graph, inputs):
    values = {k:list(v) for k,v in zip(graph['inputs'],inputs)}
    values.update({k:(v if isinstance(v,list) else [v]*8) for k,v in graph['constants'].items()})
    for row in graph['ops']:
        name, kind, operands = row[:3]; a = values[operands[0]]
        if kind == 'mul': out = [x*y for x,y in zip(a,values[operands[1]])]
        elif kind == 'add': out = [x+y for x,y in zip(a,values[operands[1]])]
        elif kind == 'negate': out = [-x for x in a]
        elif kind == 'rotate':
            shift = row[3]['steps'] % 8; out = a[shift:]+a[:shift]
        else: raise ValueError(kind)
        if not all(math.isfinite(x) for x in out): raise ValueError(f'nonfinite source at {name}')
        values[name] = out
    return [values[k] for k in graph['outputs']]


def reference_recipe(graph,case,spec,digest):
    """Small explicit local search; independent of all compiler outputs.

    Try no refresh or subsets of the current operands. Keep >=8 levels for
    subsequent exact Boot input alignment. Freeze chosen positions as a recipe.
    This conservative policy is a feasible baseline, not an optimum oracle.
    """
    plan = Plan(spec,case,digest); boots = {}
    for row in graph['ops']:
        operands = list(dict.fromkeys(v for v in row[2] if v not in graph['constants']))
        candidates = []
        for count in range(len(operands)+1):
            for names in itertools.combinations(operands,count):
                candidate = plan.clone()
                try:
                    for name in names: candidate.boot(name,row[0])
                    candidate.apply(row,graph['constants'])
                    while True:
                        state = candidate.state(candidate.bindings[row[0]])
                        if state['level'] <= spec['levels']['lower_bound'] or state['scale_log2'] - math.log2(
                            spec['context']['rns_moduli'][state['level']]) < case['min_ciphertext_scale_log2']:
                            break
                        candidate.rescale(row[0],f'after:{row[0]}')
                    if candidate.state(candidate.bindings[row[0]])['level'] < 8: continue
                except Infeasible:
                    continue
                candidates.append((candidate,names))
            if candidates: break
        if not candidates: raise Infeasible(f'large reference scheduling exhausted at {row[0]}')
        plan,names = min(candidates,key=lambda pair:(-pair[0].state(pair[0].bindings[row[0]])['level'],len(pair[0].steps)))
        if names: boots[row[0]] = list(names)
    return dict(boots_before=boots,rescale_after_each_operation=True,
                explanation='Independent Python local operand-refresh search, reserve L>=8 for exact L5/s40 Boot alignment, prefer fewer refreshes then more remaining level. Frozen decisions; no compiler output used. Feasible baseline, no global optimality proof.')


def build(destination):
    import numpy as np
    nonlinear = dict(math=math)
    qdir = POSEIDON/'scripts/qwen3_dsl/qwen3/kernels'
    sources = [source_functions(qdir/'qwen_nonlinear.py',
        ['inverse_sqrt','reciprocal','small_exp','normalize_square','normalize_square_silu'],nonlinear)]
    mpcb = dict(np=np,fint=lambda n:int(math.floor(n)))
    sources.append(source_functions(COMPILER/'python/poly/poly/MPCB.py',['GenPoly'],mpcb))
    data = COMPILER/'python/poly/poly/data'
    polynomial = mpcb['GenPoly']((data/'treeStr.txt').read_text().splitlines(),
                                (data/'coeffStr.txt').read_text().splitlines(),16)
    resnet = dict(Poly=SimpleNamespace(GenPoly=lambda:polynomial))
    sources.append(source_functions(COMPILER/'python/poly/poly/Func.py',['HE_SiLU'],resnet))
    for path in [data/'treeStr.txt',data/'coeffStr.txt',COMPILER/'python/poly/poly/models/ResNet.py',
                 qdir/'qwen_layers.py',qdir/'qwen_model.py',COMPILER/'examples/benchmarks/ResNet.py']:
        sources.append(dict(path=str(path.relative_to(POSEIDON)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            extraction='pattern topology or polynomial data; reduced dimensions documented per case'))
    # Generate the same compiler-owned profile, changing only replay dimension.
    profile_file = ROOT.parents[3]/'integrations/dacapo/generate_resnet20_manual_profiles.py'
    profile_namespace = dict(__file__=str(profile_file),Path=Path,json=json,hashlib=hashlib)
    sources.append(source_functions(profile_file,['generate'],profile_namespace))
    config,spec = profile_namespace['generate'](destination)
    config['polynomialDegree'] = 16; spec['context']['poly_degree'] = 16
    write(destination/'compiler.json',config);write(destination/'operator-spec.json',spec)
    digest = 'sha256:'+hashlib.sha256((destination/'operator-spec.json').read_bytes()).hexdigest()
    manifest = dict(name='model-large-q50',cases=[],sources=sources,description='Reduced source-derived ResNet20/Qwen arithmetic DAGs. Same compiler-generated Boot profile for reference and automatic plans. No graphs, no full-model/encrypted accuracy claim.')
    for name,builder,pattern in definitions(nonlinear,resnet['HE_SiLU']):
        graph = builder.data; directory = destination/name
        case = dict(format_version=1,case_id=name,input_names=graph['inputs'],
            inputs=[dict(kind='ciphertext',level=30,scale_log2=40,components=2,ntt=True) for _ in graph['inputs']],
            outputs=[dict(kind='ciphertext',min_level=4,min_scale_log2=40,max_scale_log2=70,components=2,ntt=True)],
            min_ciphertext_scale_log2=40,min_plaintext_scale_log2=40,min_unit_plaintext_scale_log2=30,
            numeric_tolerance=dict(abs=1e-9,rel=1e-9),test_vectors=[])
        for sample in range(3):
            inputs = [[.18*math.sin((i+1)*(sample+1)+j*.7)+.04*(j+1) for i in range(8)] for j in range(len(graph['inputs']))]
            case['test_vectors'].append(dict(inputs=inputs,outputs=evaluate(graph,inputs)))
        recipe = reference_recipe(graph,case,spec,digest)
        plan,info = materialize_recipe(graph,case,spec,digest,recipe)
        stats = verify(plan,spec,case,digest)
        info['origin'] = 'independent-python-local-search-frozen-recipe'
        pattern.update(stages=builder.stages,sources=sources,original_operation_counts=dict(Counter(row[1] for row in graph['ops'])))
        for key,obj in [('graph',graph),('case',case),('recipe',recipe),('pattern',pattern)]:write(directory/f'{key}.json',obj)
        info['inputs_sha256'] = {key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in (
            ('graph',directory/'graph.json'),('case',directory/'case.json'),('recipe',directory/'recipe.json'),
            ('pattern',directory/'pattern.json'),('operator_spec',destination/'operator-spec.json'),('generator',Path(__file__)))}
        write(directory/'reference.json',reference_metadata(plan,info,'recipe'))
        manifest['cases'].append(name)
        print(name,'ops',len(graph['ops']),'boots',stats['boot_count'],'rescale_levels',stats['rescale_levels'],flush=True)
    write(destination/'suite.json',manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    build(parser.parse_args().output_dir)
