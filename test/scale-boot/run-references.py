#!/usr/bin/env python3
"""Verify frozen references, optionally compare all compiler strategies."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

from verify import compare, read_json, verify
from reference import plan_bytes, reference_plan


def mlir_float(value):
    # MLIR requires a decimal point even when Python prints an exponent (1e-6).
    return format(float(value), ".17e")


def plan_diagnostics(plan):
    """Raw emitted counts/states, useful even when full verification fails."""
    steps = plan["initialization"] + plan["execution"] + plan["finalization"]
    values = {v["id"]: v for v in plan["values"]}
    counts = Counter(s["op"] for s in steps if s["kind"] == "compute")
    boundaries = Counter()
    for step in steps:
        if step.get("op") == "boot":
            a, b = values[step["inputs"][0]], values[step["output"]]
            boundaries[(step["attrs"]["operator_profile"], a["level"], a["scale_log2"],
                        b["level"], b["scale_log2"])] += 1
    return dict(boot_count=counts["boot"], operation_counts=dict(counts),
        boot_boundaries=[dict(profile=k[0], input_level=k[1], input_scale_log2=k[2],
                             output_level=k[3], output_scale_log2=k[4], count=n)
                         for k,n in boundaries.items()],
        note="Raw emitted metadata; not proof of plan validity. Invalid counts are excluded from comparisons.")


def earth_source(graph, case, spec):
    anchor = spec["boot_profiles"][0]["output_level"]
    input_types = [f"tensor<8x!earth.ci<0 * {anchor - x['level']}>>" for x in case["inputs"]]
    if len(set(input_types)) != 1 or spec["context"]["poly_degree"] != 16:
        raise ValueError("reference-suite frontend requires equal input levels and 8 slots")
    ct = input_types[0]
    pt = ct.replace("earth.ci", "earth.pl")
    args = ", ".join(f"%{name}: {ty}" for name, ty in zip(case["input_names"], input_types))
    output_types = ", ".join([ct] * len(graph["outputs"]))
    rows = [f"module {{ func.func @graph({args}) -> ({output_types}) {{"]
    types = dict(zip(case["input_names"], input_types))
    for name, coefficient in graph["constants"].items():
        values = coefficient if isinstance(coefficient, list) else [coefficient] * 8
        rms = math.sqrt(sum(v*v for v in values) / 8)
        literal = '[' + ', '.join(mlir_float(v) for v in values) + ']' if isinstance(coefficient, list) else mlir_float(coefficient)
        rows.append(f'%{name} = "earth.constant"() {{value = dense<{literal}> : tensor<8xf64>, '
                    f'rms_var = {mlir_float(rms)} : f64}} : () -> {pt}')
        types[name] = pt
    for row in graph["ops"]:
        name, kind, operands = row[:3]
        steps = row[3]["steps"] if len(row) == 4 and kind == "rotate" else 1
        attrs = f' <{{offset = array<i64: {steps}>}}>' if kind == "rotate" else ""
        rows.append(f'%{name} = "earth.{kind}"(' + ", ".join("%" + x for x in operands) +
                    f'){attrs} : (' + ", ".join(types[x] for x in operands) + f") -> {ct}")
        types[name] = ct
    rows.append("return " + ", ".join("%" + v for v in graph["outputs"]) + " : " +
                ", ".join(types[v] for v in graph["outputs"]))
    return "\n".join(rows + ["}}", ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path(__file__).parent / "cases/strict-small-q50")
    parser.add_argument("--hecate-opt", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strategies", nargs="+", choices=("greedy", "depth-dp", "optimized"),
                        default=["greedy", "depth-dp", "optimized"])
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--fail-on-worse", action="store_true")
    parser.add_argument("--graphs", action="store_true", help="Render offline HTML/SVG comparisons using Graphviz")
    parser.add_argument("--timeout", type=float, default=120, help="Per-compiler-run timeout in seconds")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    suite = args.suite.resolve()
    spec_path = suite / "operator-spec.json"
    spec = read_json(spec_path)
    digest = "sha256:" + hashlib.sha256(spec_path.read_bytes()).hexdigest()
    manifest = read_json(suite / "suite.json")
    names = args.cases if args.cases is not None else manifest["cases"]
    if set(names) - set(manifest["cases"]):
        raise ValueError("unknown case name")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = False
    for name in names:
        directory = suite / name
        case, graph = read_json(directory / "case.json"), read_json(directory / "graph.json")
        manual, metadata = reference_plan(directory), read_json(directory / "reference.json")
        stats = verify(manual, spec, case, digest)
        if stats["boot_count"] != metadata["boot_count"]:
            raise ValueError(f"{name}: frozen reference Boot count mismatch")
        row = dict(case=name, input_compute_ops=len(graph["ops"]), reference_boots=stats["boot_count"],
                   reference_global_optimality_proven=metadata["global_optimality_proven"],
                   manual_valid=True, algorithms={})
        depths = {name: 0 for name in graph["inputs"] + list(graph["constants"])}
        for instruction in graph["ops"]:
            value, op, operands = instruction[:3]
            depths[value] = max(depths[v] for v in operands) + (op == "mul")
        row["multiplicative_depth"] = max(depths[v] for v in graph["outputs"])
        row["reference_stats"] = stats
        row["artifacts_sha256"] = {key: hashlib.sha256((directory / f"{key}.json").read_bytes()).hexdigest()
                                   for key in ("graph", "case")}
        row["artifacts_sha256"]["manual"] = hashlib.sha256(plan_bytes(manual)).hexdigest()
        destination = args.output_dir / name
        destination.mkdir(exist_ok=True)
        (destination / "manual.json").write_bytes(plan_bytes(manual))
        (destination / "input.mlir").write_text(earth_source(graph, case, spec))
        if args.hecate_opt:
            for strategy in args.strategies:
                prefix = destination / strategy
                command = [str(args.hecate_opt.resolve()), str(destination / "input.mlir"),
                    "--dacapo", f"--boot-placement={strategy}", "--native-scale-policy=threshold",
                    f"--ckks-config={suite}/compiler.json", f"--waterline={spec['context']['default_scale_log2']}",
                    f"--bootstrap-scale={spec['boot_profiles'][0]['output_scale_log2']}",
                    "--mlir-print-op-on-diagnostic=false",
                    "--polynomial-optimizations=none", "--runtime-plan-format=json",
                    "--runtime-plan-inline-payload-max-bytes=1048576",
                    f"--runtime-plan-physical-level-operator-spec-path={spec_path}",
                    f"--runtime-plan-operator-spec-path={spec_path}",
                    f"--runtime-plan-boot-profile={spec['boot_profiles'][0]['profile_id']}",
                    f"--runtime-plan-target-id={spec['target_id']}",
                    f"--runtime-plan-context-id={spec['context']['context_id']}",
                    f"--runtime-plan-operator-spec-id={spec['spec_id']}",
                    f"--runtime-plan-operator-spec-version={spec['version']}",
                    f"--runtime-plan-operator-spec-sha256={digest}", "-o", str(prefix) + ".mlir"]
                (destination / f"{strategy}.command.json").write_text(json.dumps(command, indent=2) + "\n")
                start = time.perf_counter()
                try:
                    run = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
                except subprocess.TimeoutExpired as error:
                    stdout, stderr = error.stdout or b"", error.stderr or b""
                    (destination / f"{strategy}.log").write_bytes(stdout + stderr)
                    outcome = dict(status="timeout", compile_seconds=time.perf_counter() - start)
                    failures = True
                else:
                    elapsed = time.perf_counter() - start
                    (destination / f"{strategy}.log").write_text(run.stdout + run.stderr)
                    if run.returncode:
                        # Do not label a failed search physically infeasible: a verified reference exists.
                        outcome = dict(status="compile_failed", returncode=run.returncode,
                                       compile_seconds=elapsed, error=run.stderr[:4000])
                        failures = True
                    else:
                        automatic = read_json(Path(str(prefix) + ".graph.runtime-plan.json"))
                        report = compare(manual, automatic, spec, case, digest)
                        (destination / f"{strategy}.comparison.json").write_text(json.dumps(report, indent=2) + "\n")
                        result = report["plans"]["automatic"]
                        outcome = dict(status="valid" if result["valid"] else "invalid",
                            comparison=report["comparison"], boot_delta=report["boot_delta"], compile_seconds=elapsed)
                        outcome["plan_sha256"] = hashlib.sha256(
                            Path(str(prefix) + ".graph.runtime-plan.json").read_bytes()).hexdigest()
                        outcome["diagnostics"] = plan_diagnostics(automatic)
                        if result["valid"]:
                            outcome["stats"] = result["stats"]
                        else:
                            outcome["error"] = result["error"]
                            failures = True
                row["algorithms"][strategy] = outcome
        rows.append(row)
        print(name, "reference", row["reference_boots"],
              {s: r.get("stats", {}).get("boot_count", r["status"]) for s, r in row["algorithms"].items()}, flush=True)
    summary = dict(suite=manifest["name"], operator_spec_sha256=digest, cases=rows,
                   reference_count=len(rows), encrypted_accuracy_checked=False)
    if args.hecate_opt:
        summary["compiler"] = dict(path=str(args.hecate_opt.resolve()),
            sha256=hashlib.sha256(args.hecate_opt.read_bytes()).hexdigest(), rebuilt_by_runner=False)
        summary["comparisons"] = {s: dict(Counter(r["algorithms"][s].get("comparison", r["algorithms"][s]["status"])
                                        for r in rows)) for s in args.strategies}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.graphs:
        from draw_graphs import render_suite
        print(render_suite(suite, args.output_dir, args.output_dir / "graphs"))
    if failures:
        return 1
    return 2 if args.fail_on_worse and any(v.get("comparison") == "worse"
        for r in rows for v in r["algorithms"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
