#!/usr/bin/env python3
"""Test adapter for the C++ single-chain planner, not an MLIR pass.

Extract a private chain ending at a unit alignment, call the production core,
then materialize its returned recipe into a COPY of a RuntimePlan. No Python
scale search is performed. Payloads and rotation offsets come from the source.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import html
import json
from pathlib import Path
import subprocess
import shutil

from alignment import steps, verify_alignment_budget
from alignment_example import SUITE
from verify import HOST, decoded_unit, read_json, require, verify

SUPPORTED = {"mul_cp", "add_cp", "rotate", "negate", "rescale", "mod_switch"}


def extract(plan, unit_output, spec, case, protected=()):
    desc = {v["id"]: v for v in plan["values"]}
    producers = {s["output"]: s for s in steps(plan)}
    uses = Counter(v for s in steps(plan) for v in s.get("inputs", []))
    uses.update(plan["final_outputs"])
    slots = spec["context"]["poly_degree"] // 2

    def payload(step):
        plain = producers[step["inputs"][1]]
        require(plain["kind"] == "encode" and plain["payload"]["kind"] == "inline",
                "chain requires local inline encodings")
        values = plain["payload"]["values"]
        return values + [0.] * (slots - len(values))

    unit = producers[unit_output]
    require(unit.get("op") == "mul_cp" and payload(unit) == [1.] * slots,
            "repair root must multiply by full-slot +1")
    require(uses[unit_output] == 1, "repair root must have a single use")
    require(desc[unit_output]["components"] == 2, "only two-component chains are supported")
    chain, cursor = [], unit["inputs"][0]
    protected = set(protected) | set(plan["external_inputs"]) | set(plan["final_outputs"])
    while cursor not in protected and uses[cursor] == 1:
        op = producers.get(cursor)
        if op is None or op.get("op") not in SUPPORTED:
            break
        require(desc[cursor]["components"] == 2, "only two-component chains are supported")
        chain.append(op)
        cursor = op["inputs"][0]
    chain.reverse()
    state = lambda name: {k: desc[name][k] for k in ("level", "scale_log2")}
    operations = []
    for step in chain:
        op = dict(op=step["op"], output=state(step["output"]))
        if step["op"].endswith("_cp"):
            values = payload(step)
            op.update(plaintext_scale=desc[step["inputs"][1]]["scale_log2"],
                      unit_plaintext=values == [1.] * slots or values == [-1.] * slots)
        operations.append(op)
    request = dict(moduli=spec["context"]["rns_moduli"], input=state(cursor), target=state(unit_output),
        operations=operations, policy=dict(minimum_level=spec["levels"]["lower_bound"],
            minimum_ciphertext_scale=case["min_ciphertext_scale_log2"],
            minimum_plaintext_scale=case["min_plaintext_scale_log2"],
            maximum_unit_error=case["unit_encoding"]["max_abs_error"], capacity_margin=1.))
    return request, dict(input=cursor, chain=chain, unit=unit)


def query(driver, request):
    run = subprocess.run([str(driver.resolve())], input=json.dumps(request), text=True,
                         capture_output=True, timeout=30)
    if run.returncode not in (0, 2):
        raise RuntimeError(f"planner process failed: {run.returncode}: {run.stderr}")
    result = json.loads(run.stdout)
    require((run.returncode == 0) == (result["status"] == "success"), "driver status mismatch")
    return result


def materialize(before, context, result):
    require(result["status"] == "success", "cannot materialize a failed search")
    plan = deepcopy(before)
    desc = {v["id"]: v for v in plan["values"]}
    producers = {s["output"]: s for s in steps(plan)}
    generated, values = [], []
    current = context["input"]

    def value(state, plain=False):
        name = f"alignment.{len(values)}"
        require(name not in desc, "alignment value prefix already in use")
        v = dict(desc[context["input"]], id=name, kind="plaintext" if plain else "ciphertext",
                 components=1 if plain else 2, **state)
        values.append(v)
        return name

    for row in result["plan"]["steps"]:
        original = context["chain"][row["origin"]] if row["origin"] >= 0 else None
        operands = [current]
        if row["op"].endswith("_cp"):
            require(original is not None or row["op"] == "mul_cp", "new AddCP is not supported")
            plain = value(dict(level=row["output"]["level"], scale_log2=row["plaintext_scale"]), True)
            encoding = original if original is not None else context["unit"]
            payload = deepcopy(producers[encoding["inputs"][1]]["payload"])
            generated.append(dict(kind="encode", output=plain, payload=payload))
            operands.append(plain)
        output = value(row["output"])
        op = dict(kind="compute", op=row["op"], place=dict(HOST), inputs=operands, output=output)
        if row["op"] == "rotate":
            require(original is not None, "rotation must retain its source binding")
            op["attrs"] = deepcopy(original["attrs"])
        elif row["op"] == "rescale":
            op["attrs"] = dict(target_level=row["output"]["level"], target_scale_log2=row["output"]["scale_log2"])
        elif row["op"] == "mod_switch":
            op["attrs"] = dict(target_level=row["output"]["level"])
        generated.append(op)
        current = output
    mapping = {context["unit"]["output"]: current}
    erased = {s["output"] for s in context["chain"]} | set(mapping)
    rewritten = []
    for step in steps(plan):
        if step["output"] == context["unit"]["output"]:
            rewritten.extend(generated)
        elif step["output"] not in erased:
            if "inputs" in step:
                step["inputs"] = [mapping.get(v, v) for v in step["inputs"]]
            rewritten.append(step)
    used = {v for s in rewritten for v in s.get("inputs", [])} | set(plan["final_outputs"])
    unused = {s["output"] for s in rewritten if s["kind"] == "encode" and s["output"] not in used}
    rewritten = [s for s in rewritten if s["output"] not in unused]
    for ordinal, step in enumerate(rewritten):
        step["ordinal"] = ordinal
    plan["values"] = [v for v in plan["values"] if v["id"] not in erased | unused] + values
    plan.update(initialization=[], execution=rewritten, finalization=[])
    return plan, mapping


def solve(before, unit_output, spec, case, driver, protected=()):
    request, context = extract(before, unit_output, spec, case, protected)
    result = query(driver, request)
    if result["status"] != "success":
        return result, None, {}, request
    candidate, mapping = materialize(before, context, result)
    return result, candidate, mapping, request


def draw_automatic(directory, records, spec):
    from draw_graphs import DOT_STYLE, dot_body, plan_model
    dot = shutil.which("dot")
    if dot is None:
        raise RuntimeError("--graphs requires Graphviz dot")
    sections = []
    for name, before, candidate, case, row in records:
        if candidate is None:
            sections.append(f"<section><h2>{name}：未生成违规方案</h2><p>"
                            "原输入和要求的出口均为 L5，没有可供显式 Rescale 使用的 level 差额；"
                            "也没有可吸收调整的 MulCP。</p><pre>" +
                            html.escape(json.dumps(row["search"], indent=2)) + "</pre></section>")
            continue
        parts = ["digraph comparison {", DOT_STYLE]
        for label, plan in (("before", before), ("automatic", candidate)):
            model = plan_model(plan, case, spec, {})
            if label == "before":
                for node in model["nodes"]:
                    if node["kind"] == "align":
                        error = abs(decoded_unit(node["detail"]["input_states"][1]["scale_log2"])-1)
                        node["kind"] = "failed"
                        node["label"] += f"\nunit error: {error:.2%}"
            body, _ = dot_body(model, label)
            title = "Before: inaccurate unit" if label == "before" else "Automatic: C++ planner"
            parts.extend([f'subgraph cluster_{label} {{ label="{title}";', body, "}"])
        parts.append("}")
        path = directory / name / "comparison.dot"
        path.write_text("\n".join(parts))
        for extension in ("svg", "png"):
            subprocess.run([dot, f"-T{extension}", str(path), "-o", str(path.with_suffix("."+extension))],
                           check=True, timeout=60)
        svg = path.with_suffix(".svg").read_text()
        sections.append(f"<section><h2>{name}：{row['search']['plan']['strategy']}</h2>"
                        f"<p>出口状态保持不变；Boot=0。单位量化重放最大误差："
                        f"{row['verification']['unit_quantized_max_abs_error']:.3g}</p>"
                        "<div class='diagram'>" + svg[svg.index("<svg"):] + "</div></section>")
    (directory / "index.html").write_text("""<!doctype html><meta charset="utf-8"><title>反向对齐规划核心验收</title>
<style>body{font-family:system-ui,sans-serif;background:#f5f7fb;max-width:1400px;margin:30px auto;padding:0 20px;color:#172033}
section{background:white;border:1px solid #ddd;border-radius:10px;padding:20px;margin:24px 0}
.diagram{overflow:auto}svg{max-width:100%;height:auto}p{line-height:1.7}pre{white-space:pre-wrap}</style>
<h1>第 2 步：C++ 反向对齐规划核心</h1><p>右侧来自 C++ 自动规划，并经过独立 Python 数值与固定预算校验。
尚未接入 MLIR pass；误差模型仅覆盖单位明文量化，不模拟完整加密误差。</p>
""" + "\n".join(sections))


def run_fixtures(driver, directory, graphs=False):
    directory.mkdir(parents=True, exist_ok=True)
    spec = read_json(SUITE / "operator-spec.json")
    digest = "sha256:" + hashlib.sha256((SUITE / "operator-spec.json").read_bytes()).hexdigest()
    rows, records = [], []
    for name in read_json(SUITE / "suite.json")["cases"]:
        before = read_json(SUITE / name / "before.json")
        case = read_json(SUITE / name / "case.json")
        desc = {v["id"]: v for v in before["values"]}
        encodes = {s["output"]: s for s in steps(before) if s["kind"] == "encode"}
        units = [s for s in steps(before) if s.get("op") == "mul_cp" and
                 encodes[s["inputs"][1]]["payload"]["values"] == [1.] * 8 and
                 abs(decoded_unit(desc[s["inputs"][1]]["scale_log2"]) - 1) > case["unit_encoding"]["max_abs_error"]]
        require(len(units) == 1, "fixture must have one inaccurate unit alignment")
        result, candidate, mapping, request = solve(before, units[0]["output"], spec, case, driver)
        row = dict(case=name, search=result, before_sha256=hashlib.sha256(
            (SUITE/name/"before.json").read_bytes()).hexdigest())
        folder = directory / name
        folder.mkdir(exist_ok=True)
        (folder / "request.json").write_text(json.dumps(request, indent=2)+"\n")
        if candidate is not None:
            row["verification"] = verify(candidate, spec, case, digest)
            contract = dict(format_version=1, boundaries=[dict(name=v, before=v, after=mapping.get(v, v))
                for v in before["external_inputs"] + before["final_outputs"]])
            row["budget"] = verify_alignment_budget(before, candidate, contract)
            (folder / "automatic.json").write_text(json.dumps(candidate, indent=2)+"\n")
            (folder / "contract.json").write_text(json.dumps(contract, indent=2)+"\n")
            row["automatic_sha256"] = hashlib.sha256((folder/"automatic.json").read_bytes()).hexdigest()
        else:
            # A failed rerun must not leave a previous successful plan visible.
            for filename in ("automatic.json", "contract.json", "comparison.svg", "comparison.png", "comparison.dot"):
                (folder / filename).unlink(missing_ok=True)
        require(result["status"] == ("no-candidate" if name == "insufficient-budget" else "success"),
                f"unexpected fixture outcome: {name}: {result}")
        rows.append(row)
        records.append((name, before, candidate, case, row))
    report = dict(compiler_pass_integrated=False, encrypted_accuracy_checked=False,
                  operator_spec_sha256=digest, driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(), cases=rows)
    (directory / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
    if graphs:
        draw_automatic(directory, records, spec)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--graphs", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_fixtures(args.driver, args.output_dir, args.graphs), indent=2))
