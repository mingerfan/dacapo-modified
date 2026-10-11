#!/usr/bin/env python3
"""Check deep scheduling and Release on a large graph; compare old small plans."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import tempfile
import time


def graph(count, seed=None):
    ct = "tensor<1x!ckks.poly<2 * 40 * 5>>"
    pt = "tensor<1x!ckks.poly<1 * 40 * 5>>"
    rows = [f"module {{ func.func @scale(%arg0: {ct}) -> {ct} {{",
            f'%w = "ckks.encode"() <{{payload = dense<1.0> : tensor<1xf64>}}> : () -> {pt}']
    rng = random.Random(seed)
    available = ["%arg0"]
    for index in range(count):
        a = available[-1] if seed is None else rng.choice(available)
        op = "negatec" if seed is None else rng.choice(("rotatec", "negatec", "addcc", "addcp", "bootstrapc"))
        attrs = ""
        inputs, types = [a], [ct]
        if op == "rotatec":
            attrs = ' <{offset = array<i64: 1>}>'
        elif op == "addcc":
            inputs.append(rng.choice(available))
            types.append(ct)
        elif op == "addcp":
            inputs.append("%w")
            types.append(pt)
        rows.append(f'%v{index} = "ckks.{op}"({", ".join(inputs)}){attrs} : ({", ".join(types)}) -> {ct}')
        available.append(f"%v{index}")
    rows += [f"return {available[-1]} : {ct}", "}}"]
    return "\n".join(rows)


def compile_plan(binary, fixtures, source, directory, devices, reuse):
    directory.mkdir()
    prefix = directory / "out"
    passes = [f"assign-ckks-placement{{device-counts={devices} operator-spec={fixtures / 'placement-operator-spec.json'} boot-profile=test-boot}}",
              "materialize-ckks-communication",
              f"plan-runtime-memory{{enable-reuse={'true' if reuse else 'false'}}}",
              f"estimate-runtime-memory{{prefix={prefix} operator-spec={fixtures / 'placement-operator-spec.json'}}}",
              f"emit-runtime-plan{{plan-format=json prefix={prefix} target-id=test operator-spec-id=test operator-spec-sha256=sha256:{'0' * 64} context-id=test-context boot-profile=test-boot boot-implementation=decrypt_reencrypt}}"]
    result = subprocess.run([str(binary), str(source), f"-p=builtin.module(func.func({','.join(passes)}))",
                    "-o", str(directory / "out.mlir")], cwd=fixtures.parent.parent,
                   timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(result.stderr.decode())
    plan = json.loads((directory / "out.scale.runtime-plan.json").read_text())
    memory = json.loads((directory / "out.scale.memory.json").read_text())
    assert plan["format_version"] == 2
    assert any(op["kind"] == "release" for op in plan["initialization"] + plan["execution"])
    outputs = set(plan["final_outputs"])
    live = set(plan["external_inputs"])
    for op in plan["initialization"] + plan["execution"]:
        if op["kind"] == "release":
            assert op["value"] in live and op["value"] not in outputs
            live.remove(op["value"])
        else:
            for value in op.get("inputs", []):
                assert value in live
            if "reuse_input" in op:
                live.remove(op["inputs"][op["reuse_input"]])
            for value in op.get("outputs", [op.get("output")]):
                assert value not in live
                live.add(value)
    assert outputs <= live and live - outputs <= set(plan["external_inputs"])
    return plan, memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hecate-opt", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    args.hecate_opt = args.hecate_opt.resolve()
    args.source_dir = args.source_dir.resolve()
    if args.reference:
        args.reference = args.reference.resolve()
    fixtures = args.source_dir / "test/runtime-plan"
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="dacapo-runtime-scale-") as temporary:
        root = Path(temporary)
        for seed in range(4):
            source = root / f"random-{seed}.mlir"
            source.write_text(graph(100 + seed * 100, seed))
            for devices in (1, 4):
                for reuse in (False, True):
                    label = f"{seed}-{devices}-{reuse}"
                    plan, memory = compile_plan(args.hecate_opt, fixtures, source, root / label, devices, reuse)
                    if args.reference:
                        old_plan, old_memory = compile_plan(args.reference, fixtures, source, root / (label + "-old"), devices, reuse)
                        assert plan == old_plan, label
                        assert memory == old_memory, label
        deep = root / "deep.mlir"
        deep.write_text(graph(20000))
        plan, memory = compile_plan(args.hecate_opt, fixtures, deep, root / "deep", 4, False)
        assert sum(op.get("op") == "negate" for op in plan["execution"]) == 20000
        device_peaks = [p["peak_bytes"] for p in memory["places"] if p["place"]["kind"] == "device"]
        assert max(device_peaks) == 2 * 16 * 6 * 2 * 4  # current input and newly allocated output
        print(json.dumps({"random_cases": 16, "old_plan_comparisons": bool(args.reference),
                          "deep_chain_operations": 20000, "seconds": time.monotonic() - started}))


if __name__ == "__main__":
    main()
