#!/usr/bin/env python3
"""Build/run three manual alignment fixtures. No compiler pass or search is used."""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import subprocess

from alignment import operation_counts, verify_alignment_budget
from reference import Plan
from verify import InvalidPlan, decoded_unit, read_json, verify

ROOT = Path(__file__).resolve().parent
SUITE = ROOT / "cases/alignment-q50"
NAMES = ("absorb-mulcp", "reuse-modswitch", "insufficient-budget")


def make_case(name):
    x_level, y_level = {"absorb-mulcp": (8, 6), "reuse-modswitch": (6, 5),
                        "insufficient-budget": (5, 5)}[name]
    case = dict(format_version=1, case_id=name, input_names=["x", "y"],
        inputs=[dict(kind="ciphertext", level=level, scale_log2=scale, components=2, ntt=True)
                for level, scale in ((x_level, 40.), (y_level, 40.2))],
        outputs=[dict(kind="ciphertext", min_level=4, min_scale_log2=40,
                      max_scale_log2=41, components=2, ntt=True)],
        min_ciphertext_scale_log2=40, min_plaintext_scale_log2=40,
        min_unit_plaintext_scale_log2=30,
        numeric_tolerance=dict(abs=1e-12, rel=1e-12),
        unit_encoding=dict(max_abs_error=1e-6, output_tolerance=dict(abs=1e-9, rel=1e-9)),
        test_vectors=[])
    weight = .75 if name == "absorb-mulcp" else 1.
    for x, y in (([.125, -.25, .5, -.75, 1., -1., .3, -.4],
                  [.4, .3, -.2, .1, -.5, .6, -.7, .8]),
                 ([1., 2., -3., 4., -.5, .25, -.125, .75],
                  [-weight, -2*weight, 3*weight, -4*weight, .2, -.3, .4, -.5]),
                 ([-.2, .4, -.6, .8, -.1, .3, -.5, .7],
                  [.7, -.5, .3, -.1, .8, -.6, .4, -.2])):
        case["test_vectors"].append(dict(inputs=[x, y], outputs=[
            [weight*a+b for a, b in zip(x, y)]]))
    return case


def make_plan(name, repaired, spec, case, digest):
    """An explicit recipe, deliberately independent of a future optimizer."""
    p = Plan(spec, case, digest)
    x, y = p.external
    target = 40.2
    moduli = spec["context"]["rns_moduli"]

    def multiply(src, coefficient, plain_scale):
        a = p.state(src)
        plain = p.encode(coefficient, a["level"], plain_scale)
        return p.compute("mul_cp", [src, plain], a["level"], a["scale_log2"] + plain_scale)

    def rescale(src):
        a = p.state(src)
        level = a["level"] - 1
        scale = a["scale_log2"] - math.log2(moduli[a["level"]])
        return p.compute("rescale", [src], level, scale,
                         target_level=level, target_scale_log2=scale)

    def modswitch(src, level):
        a = p.state(src)
        return p.compute("mod_switch", [src], level, a["scale_log2"], target_level=level)

    if name == "absorb-mulcp":
        removed = math.log2(moduli[8]) + math.log2(moduli[7])
        weight_scale = target + removed - 40 if repaired else removed
        x = multiply(x, .75, weight_scale)
        x = rescale(rescale(x))
    elif name == "reuse-modswitch":
        if repaired:
            x = multiply(x, 1., target + math.log2(moduli[6]) - 40)
            x = rescale(x)
        else:
            x = modswitch(x, 5)
    elif name == "insufficient-budget" and repaired:
        # Numerically accurate but forbidden: consumes a level not in the budget.
        x = multiply(x, 1., target + math.log2(moduli[5]) - 40)
        x = rescale(x)
        y = modswitch(y, 4)
    if not repaired:
        x = multiply(x, 1., target - p.state(x)["scale_log2"])
    out = p.compute("add_cc", [x, y], p.state(x)["level"], p.state(x)["scale_log2"])
    p.bindings["out"] = out
    return p.export(["out"])


def make_fixtures():
    spec_path = ROOT / "cases/model-large-q50/operator-spec.json"
    spec_text = spec_path.read_text()
    spec = read_json(spec_path)
    digest = "sha256:" + hashlib.sha256(spec_path.read_bytes()).hexdigest()
    fixtures = {}
    for name in NAMES:
        case = make_case(name)
        before = make_plan(name, False, spec, case, digest)
        candidate = make_plan(name, True, spec, case, digest)
        boundaries = [dict(name=label, before=a, after=b) for label, a, b in zip(
            ("input-x", "input-y", "region-exit"),
            before["external_inputs"] + before["final_outputs"],
            candidate["external_inputs"] + candidate["final_outputs"])]
        fixtures[name] = dict(case=case, before=before, candidate=candidate,
            contract=dict(format_version=1, boundaries=boundaries),
            expected=dict(before_valid=False, candidate_valid=True,
                          budget_valid=name != "insufficient-budget",
                          accepted=name != "insufficient-budget"))
    return spec_text, fixtures


def write_fixtures(directory):
    spec_text, fixtures = make_fixtures()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "operator-spec.json").write_text(spec_text)
    (directory / "suite.json").write_text(json.dumps(dict(format_version=1, cases=list(NAMES)), indent=2)+"\n")
    for name, fixture in fixtures.items():
        folder = directory / name
        folder.mkdir(exist_ok=True)
        for stem, data in fixture.items():
            (folder / f"{stem}.json").write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def assess(before, candidate, contract, spec, case, digest):
    result = {"case_id": case["case_id"], "manual_recipes_only": True,
              "encrypted_accuracy_checked": False}
    for name, plan in (("before", before), ("candidate", candidate)):
        try:
            result[name] = dict(valid=True, stats=verify(plan, spec, case, digest))
        except InvalidPlan as error:
            result[name] = dict(valid=False, error=str(error))
        result[name]["operation_counts"] = operation_counts(plan)
        desc = {v["id"]: v for v in plan["values"]}
        result[name]["outputs"] = [desc[v] for v in plan["final_outputs"]]
    try:
        result["budget"] = dict(valid=True, stats=verify_alignment_budget(before, candidate, contract))
    except InvalidPlan as error:
        result["budget"] = dict(valid=False, error=str(error))
    result["accepted"] = result["candidate"]["valid"] and result["budget"]["valid"]
    return result


def draw(directory, records):
    """Render the actual fixture DAGs; Graphviz is needed only with --graphs."""
    from draw_graphs import DOT_STYLE, dot_body, plan_model
    executable = shutil.which("dot")
    if executable is None:
        raise RuntimeError("--graphs requires Graphviz dot")
    sections = []
    explanations = {
        "absorb-mulcp": "把 0.2 bit 的对齐吸收到已有权重编码；删除低精度乘 1。出口仍为 L6/s40.2。",
        "reuse-modswitch": "把 L6→L5 的 ModSwitch 换为高精度乘 1＋Rescale。出口仍为 L5/s40.2。",
        "insufficient-budget": "输入已经是 L5。高精度修复降到 L4，虽通过数值检查，仍因出口预算违规而拒绝。"}
    for name, fixture, report, spec in records:
        parts = ["digraph comparison {", DOT_STYLE,
                 'graph [label="Manual alignment fixtures | s = log2(scale)", labelloc=t];']
        for variant, title in (("before", "Before: inaccurate unit encoding"),
                               ("candidate", "Manual candidate")):
            model = plan_model(fixture[variant], fixture["case"], spec, {})
            if variant == "before":
                for node in model["nodes"]:
                    if node["kind"] == "align":
                        error = abs(decoded_unit(node["detail"]["input_states"][1]["scale_log2"]) - 1)
                        node["kind"] = "failed"
                        node["label"] += f"\nERROR: unit quantization {error:.2%}"
            elif not report["accepted"]:
                for node in model["nodes"]:
                    if node["kind"] == "output":
                        node["kind"] = "failed"
                        node["label"] += "\nREJECT: required L5, got L4"
            body, _ = dot_body(model, variant)
            parts.extend([f'subgraph cluster_{variant} {{ label="{title}";', body, "}"])
        parts.append("}")
        path = directory / f"{name}.dot"
        path.write_text("\n".join(parts))
        for extension in ("svg", "png"):
            subprocess.run([executable, f"-T{extension}", str(path), "-o",
                            str(directory / f"{name}.{extension}")], check=True, timeout=60)
        svg = (directory / f"{name}.svg").read_text()
        svg = svg[svg.index("<svg"):]
        status = "接受" if report["accepted"] else "按预期拒绝"
        sections.append(f"<section><h2>{html.escape(name)} — {status}</h2>"
                        f"<p>{explanations[name]}</p><div class='diagram'>{svg}</div>"
                        f"<details><summary>实际校验报告</summary><pre>"
                        f"{html.escape(json.dumps(report, indent=2, ensure_ascii=False))}</pre></details></section>")
    (directory / "index.html").write_text("""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Scale 对齐：第一步验收</title><style>
body{font-family:system-ui,sans-serif;background:#f5f7fb;color:#172033;max-width:1450px;margin:24px auto;padding:0 20px}
section{background:white;border:1px solid #dbe2ec;border-radius:12px;margin:24px 0;padding:20px}
p{line-height:1.7}.diagram{overflow:auto}.diagram svg{max-width:100%;height:auto}
pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer}</style>
<h1>Scale 对齐：校验器与三个核心案例</h1>
<p>所有方案为人工 recipe，尚未实现编译器 pass。红色表示原方案编码误差或候选方案预算违规。
使用真实 Q50 模数链、8-slot 明文重放；新增误差模型仅覆盖全槽 ±1，不模拟加密噪声、普通权重编码或 Rescale 舍入。</p>
""" + "\n".join(sections) + "</html>\n")


def run(suite, output_dir, graphs=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = suite / "operator-spec.json"
    spec = read_json(spec_path)
    digest = "sha256:" + hashlib.sha256(spec_path.read_bytes()).hexdigest()
    records = []
    for name in read_json(suite / "suite.json")["cases"]:
        fixture = {stem: read_json(suite / name / f"{stem}.json")
                   for stem in ("case", "before", "candidate", "contract", "expected")}
        report = assess(fixture["before"], fixture["candidate"], fixture["contract"],
                        spec, fixture["case"], digest)
        actual = dict(before_valid=report["before"]["valid"], candidate_valid=report["candidate"]["valid"],
                      budget_valid=report["budget"]["valid"], accepted=report["accepted"])
        if actual != fixture["expected"]:
            raise AssertionError((name, actual, fixture["expected"], report))
        report["expectation_met"] = True
        report["source_sha256"] = {stem: hashlib.sha256((suite/name/f"{stem}.json").read_bytes()).hexdigest()
                                   for stem in fixture}
        records.append((name, fixture, report, spec))
    summary = dict(operator_spec_sha256=digest, cases=[row[2] for row in records])
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    if graphs:
        draw(output_dir, records)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=SUITE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--write-fixtures", action="store_true",
                        help="build recipes into output-dir, without modifying the frozen suite")
    parser.add_argument("--graphs", action="store_true")
    args = parser.parse_args()
    if args.write_fixtures:
        write_fixtures(args.output_dir)
    else:
        print(json.dumps(run(args.suite, args.output_dir, args.graphs), indent=2))


if __name__ == "__main__":
    main()
