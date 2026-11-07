#!/usr/bin/env python3
"""Reproduce the strict small graphs and explicit human reference schedules."""
import argparse
import hashlib
import json
from pathlib import Path

from reference import materialize_recipe, reference_metadata
from verify import verify

ROOT = Path(__file__).resolve().parent


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


class Graph:
    def __init__(self, inputs=("x",), constants=None):
        self.data = dict(inputs=list(inputs), constants=constants or {}, ops=[], outputs=[])
        self.boots = {}

    def op(self, kind, *args):
        name = f"v{len(self.data['ops'])}"
        self.data["ops"].append([name, kind, list(args)])
        return name

    def chain(self, value, count, multiplier=None, boot_at=()):
        for i in range(count):
            if i in boot_at:
                self.boots[f"v{len(self.data['ops'])}"] = [value]
            value = self.op("mul", value, multiplier or value)
        return value


def definitions():
    # Positions are authored explicitly, before seeing compiler output.
    g = Graph(constants={"p": 0.99})
    g.data["outputs"] = [g.chain("x", 12, "p")]
    yield "cp-chain-12", g, "12 次明文乘法：消耗较多 level，但不需要 Boot 的对照。"

    g = Graph(constants={"p": 0.99})
    g.data["outputs"] = [g.chain("x", 30, "p", boot_at=(15,))]
    yield "cp-chain-30", g, "30 次明文乘法：第 16 次乘法前刷新链上的值。"

    g = Graph(constants={"p": 0.99})
    g.data["outputs"] = [g.chain("x", 42, "p", boot_at=(15, 30))]
    yield "cp-chain-42", g, "42 次明文乘法：第 16、31 次乘法前刷新，覆盖连续两次 Boot。"

    g = Graph(inputs=("x", "y"))
    g.data["outputs"] = [g.chain("x", 30, "y", boot_at=(15,))]
    yield "cc-chain-30", g, "30 次密文乘法：链值乘以保留的 y，包含长期存活输入和重线性化。"

    g = Graph()
    g.data["outputs"] = [g.chain("x", 18, boot_at=(9,))]
    yield "square-chain-18", g, "18 次平方：第 10 次平方前刷新；接近 1 的非均匀输入避免溢出和下溢。"

    g = Graph(constants={"p": 0.99, "q": 1.01})
    prefix = g.chain("x", 15, "p")
    g.boots[f"v{len(g.data['ops'])}"] = [prefix]
    left = g.chain(prefix, 12, "p")
    right = g.chain(prefix, 12, "q")
    g.data["outputs"] = [g.op("add", left, right)]
    yield "shared-prefix-fork", g, "共享 15 层乘法前缀，刷新一次后分叉；每支再乘 12 次，最后相加。"

    g = Graph(constants={"p": 0.99})
    value = g.chain("x", 30, "p", boot_at=(15,))
    value = g.op("rotate", value)
    g.data["outputs"] = [g.op("add", value, "x")]
    yield "long-residual", g, "30 次乘法后旋转并加原始 x：覆盖长短分支的 level/scale 对齐。"

    g = Graph(inputs=("x", "y"), constants={"p": 0.99, "q": 1.01})
    left = g.chain("x", 24, "p", boot_at=(15,))
    right = g.chain("y", 24, "q", boot_at=(15,))
    value = g.op("mul", left, right)
    g.data["outputs"] = [g.op("rotate", value)]
    yield "independent-branches", g, "两条独立 24 次乘法链各刷新一次，再密文相乘并旋转。"


def expected(graph, inputs):
    """Evaluate only original arithmetic, independently of any scale/Boot plan."""
    values = dict(zip(graph["inputs"], inputs))
    values.update({key: [value] * 8 for key, value in graph["constants"].items()})
    for name, kind, operands in graph["ops"]:
        a = values[operands[0]]
        if kind == "mul":
            values[name] = [x * y for x, y in zip(a, values[operands[1]])]
        elif kind == "add":
            values[name] = [x + y for x, y in zip(a, values[operands[1]])]
        elif kind == "rotate":
            values[name] = a[1:] + a[:1]
        else:
            raise ValueError(kind)
    return [values[name] for name in graph["outputs"]]


def graph_metrics(graph):
    depth = {name: 0 for name in graph["inputs"] + list(graph["constants"])}
    for name, kind, operands in graph["ops"]:
        depth[name] = max(depth[v] for v in operands) + (kind == "mul")
    return dict(compute_ops=len(graph["ops"]), multiplicative_depth=max(depth[v] for v in graph["outputs"]))


def build(destination):
    source = ROOT / "cases/native-q50"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("operator-spec.json", "compiler.json"):
        (destination / name).write_bytes((source / name).read_bytes())
    spec_path = destination / "operator-spec.json"
    spec = json.loads(spec_path.read_text())
    digest = "sha256:" + hashlib.sha256(spec_path.read_bytes()).hexdigest()
    manifest = dict(name="strict-small-q50", description="Small graphs with substantial multiplicative depth. Initial L30/s40; Boot L5/s40 -> L30/s40. Actual 50-modulus fixture; degree16 is compile-only. Hand-authored schedules, not global optimum proofs.", cases=[])
    for name, builder, explanation in definitions():
        graph = builder.data
        directory = destination / name
        state = dict(kind="ciphertext", level=30, scale_log2=40, components=2, ntt=True)
        bound = dict(kind="ciphertext", min_level=4, min_scale_log2=40, max_scale_log2=70, components=2, ntt=True)
        case = dict(format_version=1, case_id=name, input_names=graph["inputs"],
                    inputs=[state.copy() for _ in graph["inputs"]], outputs=[bound.copy() for _ in graph["outputs"]],
                    min_ciphertext_scale_log2=40, min_plaintext_scale_log2=40, min_unit_plaintext_scale_log2=30,
                    numeric_tolerance=dict(abs=1e-10, rel=1e-10), test_vectors=[])
        for sample in range(3):
            if name == "square-chain-18":
                x = [1 + (i - 3.5 + sample * 0.2) * 1e-6 for i in range(8)]
            else:
                x = [(-1 if (i + sample) % 2 else 1) * (0.2 + i * 0.08 + sample * 0.03) for i in range(8)]
            y = [0.97 + i * 0.008 + sample * 0.002 for i in range(8)]
            inputs = [x] if len(graph["inputs"]) == 1 else [x, y]
            case["test_vectors"].append(dict(inputs=inputs, outputs=expected(graph, inputs)))
        recipe = dict(boots_before=builder.boots, rescale_after_each_operation=True,
                      explanation=explanation + " 保留 s≥40 的逐操作 Rescale；Boot 前显式对齐至 L5/s40。仅保证合法，不宣称全局最优。")
        for key, obj in (("graph", graph), ("case", case), ("recipe", recipe)):
            write(directory / f"{key}.json", obj)
        plan, info = materialize_recipe(graph, case, spec, digest, recipe)
        stats = verify(plan, spec, case, digest)
        info["inputs_sha256"] = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in (
            ("graph", directory / "graph.json"), ("case", directory / "case.json"),
            ("recipe", directory / "recipe.json"), ("operator_spec", spec_path), ("generator", ROOT / "reference.py"))}
        info["graph_metrics"] = graph_metrics(graph)
        info["rescale_levels"] = stats["rescale_levels"]
        write(directory / "reference.json", reference_metadata(plan, info, "recipe"))
        manifest["cases"].append(name)
        print(name, info["graph_metrics"], 'boots', stats["boot_count"], 'rescale_levels', stats["rescale_levels"], flush=True)
    write(destination / "suite.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Explicit destination; generated reference files here are overwritten")
    build(parser.parse_args().output_dir)
