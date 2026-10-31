#!/usr/bin/env python3
"""Check reduction RS work, path depth, shared users and actual-prime states."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile


def poly(scale, level, components=2):
    return f"tensor<1x!ckks.poly<{components} * {math.ceil(scale)} * {level}>>"


def physical_source(
    moduli,
    width=2,
    drops=1,
    raw_scale=80,
    form="linear",
    shared=False,
    shared_root=False,
    reuse=False,
    rotate=False,
    fixed=False,
    unequal=False,
    different_levels=False,
    distinct_inputs=False,
):
    source = len(moduli) - 1
    target = source - drops
    scale = raw_scale
    states = []
    for i in range(drops):
        scale -= math.log2(moduli[source - i])
        states.append(scale)
    raw, low, arg = poly(raw_scale, source), poly(scale, target), poly(40, source)
    lines, products, coefficients = [], [], []
    for i in range(width):
        coefficient = (-1 if i % 2 else 1) * (i % 7 + 1) / 16
        coefficients.append(coefficient)
        # Unequal effective scales can have the same nominal integer type.
        offset = 0.01 if unequal and i == width - 1 else 0
        plain_scale = raw_scale - 40 + offset
        pt = poly(plain_scale, source, 1)
        operand = "%z" if distinct_inputs and i % 2 else "%x"
        lines += [
            f'%p{i} = "ckks.encode"() {{payload = dense<{coefficient}> : tensor<8xf64>, '
            f"runtime.scale_log2 = {plain_scale:.17e} : f64}} : () -> {pt}",
            f'%m{i} = "ckks.mulcp"({operand}, %p{i}) {{runtime.scale_log2 = {raw_scale + offset:.17e} : f64}} '
            f": ({arg}, {pt}) -> {raw}",
        ]
        previous, before = f"%m{i}", raw
        for j, actual in enumerate(states):
            # Deliberately retain equal nominal types for the unequal-scale
            # negative case; the pass must compare actual scales too.
            after = poly(actual, source - j - 1)
            attrs = "sm_fixed, " if fixed and i == 0 and j == 0 else ""
            lines.append(
                f'%r{i}_{j} = "ckks.rescalec"({previous}) '
                f"{{{attrs}runtime.scale_log2 = {actual + offset:.17e} : f64}} : ({before}) -> {after}"
            )
            previous, before = f"%r{i}_{j}", after
        products.append(previous)
    if different_levels:
        assert width == drops == 2
        level = source - 1
        actual = states[0]
        pt, raw_a, input_a = (
            poly(actual - 40, level, 1),
            poly(actual, level),
            poly(40, level),
        )
        lines[:4] = [
            f'%xm = "ckks.modswitchc"(%x) {{downFactor = 1 : i64, runtime.scale_log2 = 4.000000e+01 : f64}} : ({arg}) -> {input_a}',
            f'%p0 = "ckks.encode"() {{payload = dense<{coefficients[0]}> : tensor<8xf64>, runtime.scale_log2 = {actual - 40:.17e} : f64}} : () -> {pt}',
            f'%m0 = "ckks.mulcp"(%xm, %p0) {{runtime.scale_log2 = {actual:.17e} : f64}} : ({input_a}, {pt}) -> {raw_a}',
            f'%r0_1 = "ckks.rescalec"(%m0) {{runtime.scale_log2 = {scale:.17e} : f64}} : ({raw_a}) -> {low}',
        ]
    if rotate:
        lines.append(
            f'%rot = "ckks.rotatec"({products[0]}) '
            f"{{offset = array<i64: 1>, runtime.scale_log2 = {scale:.17e} : f64}} : ({low}) -> {low}"
        )
        products[0] = "%rot"
    counter = 0

    def add(left, right):
        nonlocal counter
        name = f"%a{counter}"
        counter += 1
        lines.append(
            f'{name} = "ckks.addcc"({left}, {right}) '
            f"{{runtime.scale_log2 = {scale:.17e} : f64}} : ({low}, {low}) -> {low}"
        )
        return name

    if reuse:
        assert width == 1
        result = add(products[0], products[0])
        coefficients *= 2
    elif form == "balanced":
        values = products
        while len(values) > 1:
            values = [
                add(values[i], values[i + 1]) if i + 1 < len(values) else values[i]
                for i in range(0, len(values), 2)
            ]
        result = values[0]
    else:
        result = products[0]
        for value in products[1:]:
            result = add(result, value)
    outputs = (
        [result] + ([products[0]] if shared else []) + (["%a0"] if shared_root else [])
    )
    types = ", ".join([low] * len(outputs))
    attrs = ", ".join(str(q) for q in moduli)
    arguments = f"%x: {arg}" + (f", %z: {arg}" if distinct_inputs else "")
    arg_scales = "4.000000e+01" + (", 4.000000e+01" if distinct_inputs else "")
    text = (
        f"module {{ func.func @reduction({arguments}) -> ({types}) "
        f"attributes {{init_level = {source} : i64, ckks.rns_moduli = array<i64: {attrs}>, "
        f"earth.arg_scales = array<f64: {arg_scales}>}} {{\n"
        + "\n".join(lines)
        + "\nreturn "
        + ", ".join(outputs)
        + f" : {types}\n}}}}\n"
    )
    return text, [
        c * (-2 if distinct_inputs and i % 2 else 1) for i, c in enumerate(coefficients)
    ]


def inspect(text):
    types, depths, outputs, ops, scales = {}, {"%arg0": 0, "%arg1": 0}, [], [], {}
    for line in text.splitlines():
        match = re.search(r'(%\w+) = "ckks\.(\w+)"\(([^)]*)\)', line)
        if match:
            name, kind, args = match.groups()
            args = re.findall(r"%\w+", args)
            metadata = [
                tuple(map(int, item))
                for item in re.findall(r"!ckks.poly<(\d+) \* (\d+) \* (\d+)>", line)
            ]
            types[name] = metadata[-1]
            if args and args[0] not in types:
                types[args[0]] = metadata[0]
            depth = max((depths[v] for v in args), default=0)
            if kind == "rescalec":
                depth += types[args[0]][2] - metadata[-1][2]
            depths[name] = depth
            scales[name] = float(re.search(r"runtime.scale_log2 = ([^ ]+)", line)[1])
            ops.append((name, kind, args, line))
        elif "return " in line:
            outputs = re.findall(r"%\w+", line)
    return dict(
        rs=sum(kind == "rescalec" for _, kind, _, _ in ops),
        muls=sum(kind == "mulcp" for _, kind, _, _ in ops),
        adds=sum(kind == "addcc" for _, kind, _, _ in ops),
        output_types=[types[v] for v in outputs],
        output_scales=[scales[v] for v in outputs],
        output_depths=[depths[v] for v in outputs],
        ops=ops,
        outputs=outputs,
    )


def replay(text, moduli):
    # Independent scalar fixed-point replay with integer rounding at RS. This
    # checks semantics/rounding, not encrypted noise or a GPU accuracy claim.
    info = inspect(text)
    values = {"%arg0": round(0.25 * 2**40), "%arg1": round(-0.5 * 2**40)}
    for name, kind, args, line in info["ops"]:
        scale = float(re.search(r"runtime.scale_log2 = ([^ ]+)", line)[1])
        if kind == "encode":
            coefficient = float(re.search(r"payload = dense<([^>]+)>", line)[1])
            values[name] = round(coefficient * 2**scale)
        elif kind == "mulcp":
            values[name] = values[args[0]] * values[args[1]]
        elif kind == "addcc":
            values[name] = values[args[0]] + values[args[1]]
        elif kind == "rotatec":
            values[name] = values[args[0]]
        elif kind == "modswitchc":
            values[name] = values[args[0]]
        elif kind == "rescalec":
            source_level = int(re.findall(r"!ckks.poly<\d+ \* \d+ \* (\d+)>", line)[0])
            q, value = moduli[source_level], values[args[0]]
            values[name] = (value + q // 2) // q
        else:
            raise AssertionError(kind)
    scales = {
        name: float(re.search(r"runtime.scale_log2 = ([^ ]+)", line)[1])
        for name, _, _, line in info["ops"]
    }
    return [values[name] / 2 ** scales[name] for name in info["outputs"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hecate-opt", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    fixture = (
        args.source_dir / "test/runtime-plan/lazy-physical-level-operator-spec.json"
    )
    uniform = [(1 << 30) - (i + 1) * 32768 + 1 for i in range(10)]
    mixed = [(1 << (30 + i % 3)) - (i + 1) * 32768 + 1 for i in range(10)]
    checks = 0
    with tempfile.TemporaryDirectory(prefix="reduction-rs-") as temporary:
        out = args.output_dir or Path(temporary)
        out.mkdir(parents=True, exist_ok=True)

        def run(name, moduli=uniform, expected=1, expensive=False, **kw):
            nonlocal checks
            source, coefficients = physical_source(moduli, **kw)
            spec = json.loads(fixture.read_text())
            spec["context"]["rns_moduli"] = moduli
            for op in ("add_cc", "rescale"):
                spec["operators"][op]["supported"] = True
                spec["operators"][op]["latency_us_by_level"] = [10] * len(moduli)
            if expensive:
                spec["operators"]["add_cc"]["latency_us_by_level"][-1] = 1000
            (out / "spec.json").write_text(json.dumps(spec))
            input_file, output = out / (name + ".input.mlir"), out / (name + ".mlir")
            input_file.write_text(source)
            pipeline = (
                "builtin.module(func.func(coalesce-reduction-rescales{"
                f"operator-spec={out}/spec.json" + "}))"
            )
            command = [
                str(args.hecate_opt.resolve()),
                str(input_file),
                f"--ckks-config={args.source_dir}/config.json",
                f"--pass-pipeline={pipeline}",
                "-o",
                str(output),
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, (name, result.stderr)
            text = output.read_text()
            normalized_source = source.replace("%x", "%arg0").replace("%z", "%arg1")
            before, after = inspect(normalized_source), inspect(text)
            assert after["rs"] == expected, (name, after["rs"], expected, text)
            assert before["muls"] == after["muls"] and before["adds"] == after["adds"]
            assert before["output_types"] == after["output_types"]
            assert before["output_depths"] == after["output_depths"]
            assert before["output_scales"] == after["output_scales"]
            if not kw.get("unequal"):
                a, b = (
                    replay(normalized_source, moduli),
                    replay(text, moduli),
                )
                assert max(abs(x - y) for x, y in zip(a, b)) < 1e-8, (name, a, b)
                assert abs(b[0] - 0.25 * sum(coefficients)) < 1e-8, (name, b)
            # The second application must not progressively rescan/rewrite the
            # same reduction or add a further RS.
            command[1] = str(output)
            command[-1] = str(out / "again.mlir")
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, result.stderr
            assert (out / "again.mlir").read_text() == text, name
            print(
                name,
                "RS",
                before["rs"],
                "->",
                after["rs"],
                "path Q",
                after["output_depths"],
                flush=True,
            )
            checks += 1

        run("two-products")
        run("two-distinct-inputs", distinct_inputs=True)
        run("long-sum", width=64)
        run("balanced", width=32, form="balanced")
        run("two-rescales", width=8, drops=2, raw_scale=110, expected=2)
        run("mixed-primes", moduli=mixed, width=16, drops=2, raw_scale=110, expected=2)
        run("shared-leaf", shared=True, expected=2)
        run(
            "shared-subtree-exit",
            width=4,
            form="balanced",
            shared_root=True,
            expected=2,
        )
        run("reuse-one-product", width=1, reuse=True, expected=1)
        run("rotation-barrier", rotate=True, expected=2)
        run("fixed-rescale", fixed=True, expected=2)
        run("unequal-actual-scales", unequal=True, raw_scale=80.1, expected=2)
        run(
            "different-raw-levels",
            drops=2,
            raw_scale=110,
            different_levels=True,
            expected=3,
        )
        run("unprofitable-adds", expensive=True, expected=2)
        # Capacity carries split a four-term tree into two safe groups.
        exact = [1 << 30] * 3
        run("capacity-boundary", moduli=exact, width=2, raw_scale=89, expected=2)
        run("capacity-split", moduli=exact, width=4, raw_scale=88, expected=2)

        # A malformed cost model is an error, not an unnoticed alternative plan.
        spec = json.loads((out / "spec.json").read_text())
        del spec["operators"]["add_cc"]["latency_us_by_level"]
        (out / "spec.json").write_text(json.dumps(spec))
        result = subprocess.run(
            [
                str(args.hecate_opt.resolve()),
                str(out / "capacity-split.input.mlir"),
                f"--ckks-config={args.source_dir}/config.json",
                "--pass-pipeline=builtin.module(func.func(coalesce-reduction-rescales{"
                f"operator-spec={out}/spec.json" + "}))",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0 and "latency tables" in result.stderr, (
            result.stderr
        )
        checks += 1

        # Exercise the default pipeline on each Boot policy, including two RS
        # per product and mixed-width primes. IDs may change; output states and
        # the critical-path Q must not.
        ct, pt = "tensor<1x!earth.ci<0 * 0>>", "tensor<1x!earth.pl<0 * 0>>"
        for chain_name, chain in [("uniform", uniform * 5), ("mixed", mixed * 5)]:
            spec = json.loads(fixture.read_text())
            spec["rescale_mode"] = "native"
            spec["context"].update(
                rns_moduli=chain,
                rns_moduli_log2=[q.bit_length() for q in chain],
                default_scale_log2=40,
                max_modulus_log2=max(q.bit_length() for q in chain),
            )
            for name, entry in spec["operators"].items():
                entry["supported"] = True
                if name != "boot":
                    entry["latency_us_by_level"] = [10] * len(chain)
            spec["boot_profiles"] = [
                dict(
                    profile_id="reduction40",
                    implementation="native",
                    input_level_min=4,
                    input_level_max=49,
                    input_components=2,
                    output_level=15,
                    output_scale_log2=40,
                    output_components=2,
                    latency_us_by_input_level=[85000] * 50,
                    noise_by_input_level=None,
                    host_requirements=dict(
                        needs_secret_key=False, needs_host_compute=False
                    ),
                )
            ]
            (out / "pipeline-spec.json").write_text(json.dumps(spec))
            config = json.loads((args.source_dir / "config.json").read_text())
            config.update(
                polynomialDegree=16,
                rescalingFactor=30,
                bootstrapLevelLowerBound=1,
                bootstrapLevelUpperBound=12,
                levelLowerBound=1,
                levelUpperBound=12,
            )
            (out / "config.json").write_text(json.dumps(config))
            for seeds in (0, 2):
                lines = [
                    f"module {{ func.func @reduce(%x: {ct}) -> {ct} {{",
                    f'%c = "earth.constant"() {{value = dense<1.125> : tensor<8xf64>, rms_var = 1.125 : f64}} : () -> {pt}',
                ]
                value = "%x"
                for i in range(seeds):
                    lines.append(
                        f'%seed{i} = "earth.mul"({value}, %c) : ({ct}, {pt}) -> {ct}'
                    )
                    value = f"%seed{i}"
                for i in range(8):
                    coefficient = (i + 1) / 16
                    lines += [
                        f'%p{i} = "earth.constant"() {{value = dense<{coefficient}> : tensor<8xf64>, rms_var = 1.0 : f64}} : () -> {pt}',
                        f'%m{i} = "earth.mul"({value}, %p{i}) : ({ct}, {pt}) -> {ct}',
                    ]
                value = "%m0"
                for i in range(1, 8):
                    lines.append(
                        f'%sum{i} = "earth.add"({value}, %m{i}) : ({ct}, {ct}) -> {ct}'
                    )
                    value = f"%sum{i}"
                lines += [f"return {value} : {ct}", "}}"]
                (out / "earth.mlir").write_text("\n".join(lines))
                for strategy in ("greedy", "depth-dp", "optimized"):
                    plans = []
                    for enabled in (False, True):
                        dest = out / f"{chain_name}-{seeds}-{strategy}-{enabled}.mlir"
                        command = [
                            str(args.hecate_opt.resolve()),
                            str(out / "earth.mlir"),
                            "--dacapo",
                            f"--ckks-config={out}/config.json",
                            "--waterline=40",
                            "--bootstrap-scale=40",
                            "--polynomial-optimizations=none",
                            f"--boot-placement={strategy}",
                            f"--runtime-plan-coalesce-rescales={str(enabled).lower()}",
                            "--runtime-plan-format=json",
                            "--runtime-plan-boot-profile=reduction40",
                            f"--runtime-plan-physical-level-operator-spec-path={out}/pipeline-spec.json",
                            "--runtime-plan-levels-per-logical-level=1",
                            "--runtime-plan-device-counts=1",
                            "--runtime-plan-device-count=1",
                            f"--runtime-plan-operator-spec-path={out}/pipeline-spec.json",
                            "--runtime-plan-target-id=lazy-physical-level-test",
                            "--runtime-plan-context-id=lazy-physical-level-test",
                            "--runtime-plan-operator-spec-id=lazy-physical-level-test-v1",
                            "--runtime-plan-operator-spec-version=1",
                            "--runtime-plan-operator-spec-sha256=sha256:"
                            + hashlib.sha256(
                                (out / "pipeline-spec.json").read_bytes()
                            ).hexdigest(),
                            "-o",
                            str(dest),
                        ]
                        result = subprocess.run(
                            command, capture_output=True, text=True, timeout=90
                        )
                        assert result.returncode == 0, result.stderr[-6000:]
                        plans.append(
                            json.loads(
                                dest.with_name(
                                    dest.stem + ".reduce.runtime-plan.json"
                                ).read_text()
                            )
                        )
                    states, counts, depths = [], [], []
                    for plan in plans:
                        values = {v["id"]: v for v in plan["values"]}
                        states.append(
                            [
                                (values[v]["level"], values[v]["scale_log2"])
                                for v in plan["final_outputs"]
                            ]
                        )
                        steps = [
                            s for s in plan["execution"] if s.get("op") == "rescale"
                        ]
                        counts.append(len(steps))
                        assert not any(s.get("op") == "boot" for s in plan["execution"])
                        depth = {v: 0 for v in plan["external_inputs"]}
                        for step in plan["initialization"] + plan["execution"]:
                            if "output" not in step:
                                continue
                            value = max(
                                (depth.get(v, 0) for v in step.get("inputs", [])),
                                default=0,
                            )
                            if step.get("op") == "rescale":
                                value += (
                                    values[step["inputs"][0]]["level"]
                                    - values[step["output"]]["level"]
                                )
                            depth[step["output"]] = value
                        depths.append([depth[v] for v in plan["final_outputs"]])
                    assert states[0] == states[1] and depths[0] == depths[1], (
                        states,
                        depths,
                    )
                    # The common seed keeps its RS. The eight reduction terms
                    # contribute one shared RS sequence instead of eight.
                    assert counts[0] > counts[1] and (counts[0] - counts[1]) % 7 == 0, (
                        counts
                    )
                    print(
                        chain_name,
                        seeds,
                        strategy,
                        "pipeline RS",
                        counts,
                        "path Q",
                        depths[1],
                        flush=True,
                    )
                    checks += 1
        # A fixed Earth multiply must still be fixed after lowering to CKKS.
        earth = (out / "earth.mlir").read_text()
        earth = re.sub(r'(%m0 = "earth.mul"\([^\n]+?\)) :', r"\1 {sm_fixed} :", earth)
        assert "{sm_fixed}" in earth
        (out / "earth.mlir").write_text(earth)
        fixed_counts = []
        for enabled in (False, True):
            fixed_command = [a for a in command]
            index = next(
                i
                for i, a in enumerate(fixed_command)
                if a.startswith("--runtime-plan-coalesce-rescales=")
            )
            fixed_command[index] = (
                f"--runtime-plan-coalesce-rescales={str(enabled).lower()}"
            )
            dest = out / f"fixed-earth-{enabled}.mlir"
            fixed_command[-1] = str(dest)
            result = subprocess.run(
                fixed_command, capture_output=True, text=True, timeout=90
            )
            assert result.returncode == 0, result.stderr
            assert "sm_fixed" in dest.read_text()
            plan = json.loads(
                dest.with_name(dest.stem + ".reduce.runtime-plan.json").read_text()
            )
            fixed_counts.append(
                sum(s.get("op") == "rescale" for s in plan["execution"])
            )
        assert fixed_counts[0] == fixed_counts[1], fixed_counts
        checks += 1
    print(checks, "reduction RS cases passed")


if __name__ == "__main__":
    main()
