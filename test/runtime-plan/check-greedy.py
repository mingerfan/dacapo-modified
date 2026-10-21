#!/usr/bin/env python3
"""Check lazy/depth-DP placement, fanout, shared inputs, and residuals."""
import argparse
import json
import math
import random
import subprocess
import tempfile
from pathlib import Path


def source(depth, cipher_mul=False):
    cipher = "tensor<1x!earth.ci<0 * 0>>"
    plain = "tensor<1x!earth.pl<0 * 0>>"
    lines = [f"module {{ func.func @chain(%x: {cipher}) -> {cipher} {{",
             f'%p = "earth.constant"() {{value = dense<1.01> : tensor<1xf64>, '
             f'rms_var = 1.01 : f64}} : () -> {plain}']
    previous = "%x"
    for index in range(depth):
        rhs, rhs_type = ("%x", cipher) if cipher_mul else ("%p", plain)
        lines.append(f'%m{index} = "earth.mul"({previous}, {rhs}) : '
                     f'({cipher}, {rhs_type}) -> {cipher}')
        previous = f"%m{index}"
    lines.extend([f'%y = "earth.add"({previous}, %x) : '
                  f'({cipher}, {cipher}) -> {cipher}',
                  f"return %y : {cipher}", "}}"])
    return "\n".join(lines)


def fanout_source(width):
    cipher = "tensor<1x!earth.ci<0 * 0>>"
    plain = "tensor<1x!earth.pl<0 * 0>>"
    lines = [f"module {{ func.func @chain(%x: {cipher}) -> {cipher} {{",
             f'%p = "earth.constant"() {{value = dense<1.01> : tensor<1xf64>, '
             f'rms_var = 1.01 : f64}} : () -> {plain}',
             f'%a = "earth.mul"(%x, %p) : ({cipher}, {plain}) -> {cipher}',
             f'%b = "earth.mul"(%a, %p) : ({cipher}, {plain}) -> {cipher}']
    previous = "%x"
    for index in range(width):
        coefficient = 1.02 + index / 100
        lines.extend([
            f'%p{index} = "earth.constant"() {{value = dense<{coefficient}> : '
            f'tensor<1xf64>, rms_var = {coefficient} : f64}} : () -> {plain}',
            f'%r{index} = "earth.rotate"(%b) <{{offset = array<i64: {index + 1}>}}> : '
            f'({cipher}) -> {cipher}',
            f'%c{index} = "earth.mul"(%r{index}, %p{index}) : '
            f'({cipher}, {plain}) -> {cipher}',
            f'%d{index} = "earth.mul"(%c{index}, %p) : '
            f'({cipher}, {plain}) -> {cipher}',
            f'%s{index} = "earth.add"({previous}, %d{index}) : '
            f'({cipher}, {cipher}) -> {cipher}'])
        previous = f"%s{index}"
    lines.extend([f"return {previous} : {cipher}", "}}"])
    return "\n".join(lines)


def mixed_source(seed):
    generator = random.Random(seed)
    cipher = "tensor<1x!earth.ci<0 * 0>>"
    plain = "tensor<1x!earth.pl<0 * 0>>"
    lines = [f"module {{ func.func @chain(%x: {cipher}) -> {cipher} {{",
             f'%p = "earth.constant"() {{value = dense<1.01> : tensor<1xf64>, '
             f'rms_var = 1.01 : f64}} : () -> {plain}']
    names, decoded = ["%x"], [0.25]
    for index in range(36):
        if index % 3 == 0:
            rhs = generator.randrange(len(names))
            op, rhs_name, rhs_type = "add", names[rhs], cipher
            value = decoded[-1] + decoded[rhs]
        elif index % 5 == 0:
            op, rhs_name, rhs_type = "mul", "%x", cipher
            value = decoded[-1] * 0.25
        else:
            op, rhs_name, rhs_type = "mul", "%p", plain
            value = decoded[-1] * 1.01
        lines.append(f'%v{index} = "earth.{op}"({names[-1]}, {rhs_name}) : '
                     f'({cipher}, {rhs_type}) -> {cipher}')
        names.append(f"%v{index}")
        decoded.append(value)
    lines.extend([f"return {names[-1]} : {cipher}", "}}"])
    return "\n".join(lines), decoded[-1]


def reduction_source(width):
    cipher = "tensor<1x!earth.ci<0 * 0>>"
    plain = "tensor<1x!earth.pl<0 * 0>>"
    lines = [f"module {{ func.func @chain(%x: {cipher}) -> {cipher} {{",
             f'%p = "earth.constant"() {{value = dense<1.01> : tensor<1xf64>, '
             f'rms_var = 1.01 : f64}} : () -> {plain}',
             f'%a = "earth.mul"(%x, %p) : ({cipher}, {plain}) -> {cipher}',
             f'%b = "earth.mul"(%a, %p) : ({cipher}, {plain}) -> {cipher}']
    previous = "%x"
    for index in range(width):
        coefficient = 1.02 + index / 100
        lines.extend([
            f'%p{index} = "earth.constant"() {{value = dense<{coefficient}> : '
            f'tensor<1xf64>, rms_var = {coefficient} : f64}} : () -> {plain}',
            f'%c{index} = "earth.mul"(%b, %p{index}) : '
            f'({cipher}, {plain}) -> {cipher}',
            f'%s{index} = "earth.add"({previous}, %c{index}) : '
            f'({cipher}, {cipher}) -> {cipher}'])
        previous = f"%s{index}"
    lines.extend([f'%y = "earth.mul"({previous}, %p) : '
                  f'({cipher}, {plain}) -> {cipher}', f"return %y : {cipher}", "}}"])
    return "\n".join(lines)


def mask_source(length, sign):
    cipher = "tensor<1x!earth.ci<0 * 0>>"
    plain = "tensor<1x!earth.pl<0 * 0>>"
    return (f"module {{ func.func @chain(%x: {cipher}) -> {cipher} {{\n"
            f'%p = "earth.constant"() {{value = dense<{float(sign)}> : tensor<{length}xf64>, '
            f'rms_var = 1.0 : f64}} : () -> {plain}\n'
            f'%y = "earth.mul"(%x, %p) : ({cipher}, {plain}) -> {cipher}\n'
            f'return %y : {cipher}\n}}}}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hecate-opt", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="dacapo-greedy-") as directory:
        temp = Path(directory)
        config = json.loads((args.source_dir / "config.json").read_text())
        config.update(polynomialDegree=16, rescalingFactor=120,
                      bootstrapLevelLowerBound=1, bootstrapLevelUpperBound=4,
                      levelLowerBound=1, levelUpperBound=4)
        profile = temp / "config.json"
        profile.write_text(json.dumps(config))
        cases = [
            ("shallow", source(1), 0.25 * 1.01 + 0.25, False),
            ("deep", source(12), 0.25 * 1.01 ** 12 + 0.25, True),
            ("cipher", source(12, True), 0.25 ** 13 + 0.25, True),
            ("fanout", fanout_source(8),
             0.25 + 0.25 * 1.01 ** 3 * sum(1.02 + i / 100 for i in range(8)), False),
            ("reduction", reduction_source(8),
             (0.25 + 0.25 * 1.01 ** 2 * sum(1.02 + i / 100 for i in range(8))) * 1.01,
             False),
            ("exhausted_input", source(1).replace("ci<0 * 0>", "ci<0 * 3>")
             .replace("pl<0 * 0>", "pl<0 * 3>"),
             0.25 * 1.01 + 0.25, False),
        ]
        for seed in range(5):
            text, expected = mixed_source(seed)
            cases.append((f"mixed{seed}", text, expected, True))
        fanout_boots = {}
        for strategy, name, text, expected, needs_boot in (
                (strategy, *case) for strategy in ("greedy", "depth-dp") for case in cases):
            traced, output = temp / "input.mlir", temp / "result.mlir"
            traced.write_text(text)
            command = [str(args.hecate_opt), "--dacapo", f"--boot-placement={strategy}",
                       f"--ckks-config={profile}", "--waterline=40",
                       "--mlir-disable-threading", "--mlir-timing", "--runtime-plan-id=1", "--runtime-plan-format=json",
                       "--runtime-plan-target-id=test", "--runtime-plan-context-id=test",
                       "--runtime-plan-operator-spec-id=test", "--runtime-plan-operator-spec-version=1",
                       "--runtime-plan-operator-spec-sha256=sha256:" + "0" * 64,
                       "--runtime-plan-boot-profile=test", "--runtime-plan-boot-implementation=native",
                       str(traced), "-o", str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            assert result.returncode == 0, (strategy, name, result.stderr)
            assert "GreedyBootstrapPlacement" in result.stderr
            assert "BypassDetection" not in result.stderr
            assert "DaCapoPlanner" not in result.stderr
            if strategy == "depth-dp":
                assert "Depth-DP bootstrap boundaries" in result.stdout
            if name == "shallow" and strategy == "greedy":
                optimized = subprocess.run(
                    [arg for arg in command if arg != "--boot-placement=greedy"],
                    capture_output=True, text=True, check=True)
                assert "BypassDetection" in optimized.stderr
                assert "GreedyBootstrapPlacement" not in optimized.stderr
                # Restore the greedy artifacts for the checks below.
                subprocess.run(command, capture_output=True, text=True, check=True)
            plan = json.loads(output.with_name(output.stem + ".chain.runtime-plan.json").read_text())
            descriptors = {value["id"]: value for value in plan["values"]}
            decoded = {plan["external_inputs"][0]: 0.25}
            for step in plan["initialization"]:
                decoded[step["output"]] = step["payload"]["values"][0]
            boots, boot_values = 0, []
            for step in plan["execution"]:
                operands = step["inputs"]
                assert all(value in decoded for value in operands)
                before = [descriptors[value] for value in operands]
                after = descriptors[step["output"]]
                op = step["op"]
                if op in ("mul_cp", "mul_cc", "add_cp", "add_cc"):
                    assert before[0]["level"] == before[1]["level"]
                    if op.startswith("mul"):
                        assert after["scale_log2"] == sum(v["scale_log2"] for v in before)
                        value = decoded[operands[0]] * decoded[operands[1]]
                    else:
                        assert before[0]["scale_log2"] == before[1]["scale_log2"]
                        value = decoded[operands[0]] + decoded[operands[1]]
                else:
                    assert op in ("rescale", "mod_switch", "relinearize", "boot", "rotate")
                    value = decoded[operands[0]]
                    if op == "boot":
                        boots += 1
                        boot_values.append(value)
                        assert before[0]["level"] >= 1
                        assert after["level"] == 4 and after["scale_log2"] == 120
                decoded[step["output"]] = value
                assert 0 <= after["level"] <= 4
            assert (boots > 0) == needs_boot, (strategy, name, boots)
            assert math.isclose(decoded[plan["final_outputs"][0]], expected, rel_tol=1e-12)
            if name == "fanout":
                fanout_boots[strategy] = boots
            if name == "reduction" and strategy == "depth-dp":
                # Threshold rescaling fits this reduction without a refresh.
                assert boots == 0

        assert fanout_boots["depth-dp"] == 0, fanout_boots
        assert fanout_boots["depth-dp"] <= fanout_boots["greedy"], fanout_boots

        # Frontend-style canonicalization has no selected runtime slot capacity.
        traced.write_text(mask_source(8, 1))
        result = subprocess.run([str(args.hecate_opt), "--canonicalize",
                                 f"--ckks-config={profile}", str(traced), "-o", str(output)],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "earth.mul" in output.read_text()

        for strategy in ("greedy", "depth-dp", "optimized"):
            command = [f"--boot-placement={strategy}" if arg.startswith("--boot-placement=")
                       else arg for arg in command]
            for length, sign in ((3, 1), (3, -1), (8, 1), (8, -1)):
                traced.write_text(mask_source(length, sign))
                result = subprocess.run(command, capture_output=True, text=True)
                assert result.returncode == 0, result.stderr
                plan = json.loads(output.with_name(output.stem + ".chain.runtime-plan.json").read_text())
                original = [0.25 + index for index in range(8)]
                decoded = {plan["external_inputs"][0]: original}
                for step in plan["initialization"]:
                    payload = step["payload"]["values"]
                    decoded[step["output"]] = payload + [0.0] * (8 - len(payload))
                for step in plan["execution"]:
                    operands = [decoded[value] for value in step["inputs"]]
                    if step["op"] == "mul_cp":
                        value = [a * b for a, b in zip(*operands)]
                    elif step["op"] == "negate":
                        value = [-a for a in operands[0]]
                    else:
                        assert step["op"] in ("mod_switch", "rescale", "boot")
                        value = operands[0]
                    decoded[step["output"]] = value
                expected = [sign * value if index < length else 0.0
                            for index, value in enumerate(original)]
                assert decoded[plan["final_outputs"][0]] == expected, (strategy, length, sign)

        config["bootstrapLevelUpperBound"] = 1
        profile.write_text(json.dumps(config))
        command = ["--boot-placement=depth-dp" if arg.startswith("--boot-placement=")
                   else arg for arg in command]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode != 0
        assert "usable bootstrap budget" in result.stderr
    print("Lazy/depth-DP chains, residuals, shared inputs, fanout, and budget checks passed")


if __name__ == "__main__":
    main()
