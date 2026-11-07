#!/usr/bin/env python3
"""Independent bounded search that materializes reference RuntimePlans.

No compiler calls or compiler-plan inputs. Search decisions are optional input
refreshes and rescale locations; coefficient encoding is fixed at 40 bits.
The retained beam is not a proof of global optimality.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from verify import HOST, read_json, verify


class Infeasible(ValueError):
    """A particular search transition violates the common contract."""


class Plan:
    def __init__(self, spec, case, digest):
        self.spec, self.case, self.digest = spec, case, digest
        self.values, self.steps, self.events = [], [], []
        self.bindings = {}
        self.boots = 0
        self.external = []
        for name, state in zip(case["input_names"], case["inputs"]):
            value = self.value(state["level"], state["scale_log2"], state["components"])
            self.bindings[name] = value
            self.external.append(value)

    def clone(self):
        other = copy.copy(self)
        other.values, other.steps, other.events = self.values[:], self.steps[:], self.events[:]
        other.bindings = self.bindings.copy()
        return other

    def state(self, value):
        return self.values[int(value)]

    def value(self, level, scale, components=2, kind="ciphertext"):
        lo, hi = self.spec["levels"]["lower_bound"], self.spec["levels"]["upper_bound"]
        capacity = sum(math.log2(q) for q in self.spec["context"]["rns_moduli"][:level+1])
        if not (lo <= level <= hi and 0 <= scale < capacity):
            raise Infeasible("value outside modulus capacity")
        if kind == "ciphertext" and scale < self.case["min_ciphertext_scale_log2"] - 1e-8:
            raise Infeasible("ciphertext scale below floor")
        name = str(len(self.values))
        self.values.append(dict(id=name, kind=kind, level=level, scale_log2=scale,
                                components=components, ntt=True, place=dict(HOST),
                                context=self.spec["context"]["context_id"]))
        return name

    def compute(self, op, operands, level, scale, components=2, **attrs):
        out = self.value(level, scale, components)
        step = dict(ordinal=len(self.steps), kind="compute", op=op,
                    inputs=operands, output=out, place=dict(HOST))
        if attrs:
            step["attrs"] = attrs
        self.steps.append(step)
        return out

    def encode(self, coefficient, level, scale):
        out = self.value(level, scale, 1, "plaintext")
        payload = coefficient if isinstance(coefficient, list) else [coefficient] * (self.spec["context"]["poly_degree"] // 2)
        self.steps.append(dict(ordinal=len(self.steps), kind="encode", output=out,
            payload=dict(kind="inline", values=payload)))
        return out

    def change(self, original, replacement):
        # All subsequent uses may share the refreshed/rescaled SSA value.
        for name, value in list(self.bindings.items()):
            if value == original:
                self.bindings[name] = replacement

    def record(self, at, op, before, after):
        self.events.append(dict(at=at, op=op, input=before, output=after,
            before={k: self.state(before)[k] for k in ("level", "scale_log2")},
            after={k: self.state(after)[k] for k in ("level", "scale_log2")}))

    def boot(self, name, at):
        old = self.bindings[name]
        a, profile = self.state(old), self.spec["boot_profiles"][0]
        if "input_scale_log2" in profile and abs(a["scale_log2"] - profile["input_scale_log2"]) > 1e-8:
            # Align by a real encoded-one multiplication and explicit division
            # by Q, reserving the declared encoding precision and Boot level.
            target = profile["input_scale_log2"]
            floor = self.case["min_unit_plaintext_scale_log2"]
            removed = 0.
            found = False
            for count in range(1, a["level"] - profile["input_level_min"] + 1):
                removed += math.log2(self.spec["context"]["rns_moduli"][a["level"] - count + 1])
                encoding = target - a["scale_log2"] + removed
                if encoding >= floor:
                    found = True
                    break
            if not found:
                raise Infeasible("insufficient Q to align Boot input")
            unit = self.encode(1., a["level"], encoding)
            aligned = self.compute("mul_cp", [old, unit], a["level"], a["scale_log2"] + encoding)
            self.change(old, aligned)
            self.record(at, "align_boot_input", old, aligned)
            for _ in range(count):
                self.rescale(name, at)
            old = self.bindings[name]
            a = self.state(old)
        if a["level"] > profile["input_level_max"]:
            lowered = self.compute("mod_switch", [old], profile["input_level_max"],
                                   a["scale_log2"], target_level=profile["input_level_max"])
            self.change(old, lowered)
            self.record(at, "mod_switch", old, lowered)
            old, a = lowered, self.state(lowered)
        if not profile["input_level_min"] <= a["level"] <= profile["input_level_max"]:
            raise Infeasible("Boot input level")
        out = self.compute("boot", [old], profile["output_level"], profile["output_scale_log2"],
            target_level=profile["output_level"], target_scale_log2=profile["output_scale_log2"],
            target_components=2, operator_profile=profile["profile_id"], implementation=profile["implementation"])
        self.change(old, out)
        self.boots += 1
        self.record(at, "boot", old, out)

    def rescale(self, name, at):
        old = self.bindings[name]
        a = self.state(old)
        level = a["level"] - 1
        scale = a["scale_log2"] - math.log2(self.spec["context"]["rns_moduli"][a["level"]])
        out = self.compute("rescale", [old], level, scale,
                           target_level=level, target_scale_log2=scale)
        self.change(old, out)
        self.record(at, "rescale", old, out)

    def apply(self, row, constants):
        name, kind, inputs = row[:3]
        attrs = row[3] if len(row) == 4 else {}
        ct = [v for v in inputs if v not in constants]
        ids = [self.bindings[v] for v in ct]
        level = min(self.state(v)["level"] for v in ids)
        aligned = {}
        for value in ids:
            a = self.state(value)
            aligned[value] = value if a["level"] == level else self.compute(
                "mod_switch", [value], level, a["scale_log2"], target_level=level)
            if aligned[value] != value:
                self.record(name, "mod_switch", value, aligned[value])
        ids = [aligned[v] for v in ids]
        if kind == "mul":
            if len(ct) == 1:
                if len(inputs) != 2 or inputs[1] not in constants:
                    raise ValueError("reference format requires ciphertext * plaintext order")
                pt = self.encode(constants[inputs[1]], level, 40.)
                scale = self.state(ids[0])["scale_log2"] + 40.
                out = self.compute("mul_cp", [ids[0], pt], level, scale)
            else:
                scale = sum(self.state(v)["scale_log2"] for v in ids)
                product = self.compute("mul_cc", ids, level, scale, 3)
                out = self.compute("relinearize", [product], level, scale)
        elif kind == "add":
            if len(ct) == 1 and len(inputs) == 2 and inputs[1] in constants:
                scale = self.state(ids[0])["scale_log2"]
                pt = self.encode(constants[inputs[1]], level, scale)
                self.bindings[name] = self.compute("add_cp", [ids[0], pt], level, scale)
                return
            if len(ct) != 2:
                raise ValueError("reference add requires two ciphertexts")
            scale = max(self.state(v)["scale_log2"] for v in ids)
            floor = self.case["min_unit_plaintext_scale_log2"]
            if any(1e-8 < scale - self.state(v)["scale_log2"] < floor for v in ids):
                scale += floor
            raised = {}
            for value in ids:
                delta = scale - self.state(value)["scale_log2"]
                raised[value] = value
                if delta > 1e-8:
                    if delta < self.case["min_unit_plaintext_scale_log2"]:
                        raise Infeasible("alignment below unit-encoding floor")
                    unit = self.encode(1., level, delta)
                    raised[value] = self.compute("mul_cp", [value, unit], level, scale)
                    self.record(name, "align_scale", value, raised[value])
            out = self.compute("add_cc", [raised[v] for v in ids], level, scale)
        elif kind in ("rotate", "negate"):
            attrs = {"steps": attrs.get("steps", 1)} if kind == "rotate" else {}
            out = self.compute(kind, ids, level, self.state(ids[0])["scale_log2"], **attrs)
        else:
            raise ValueError(f"unsupported reference operation: {kind}")
        self.bindings[name] = out

    def export(self, outputs):
        return dict(format_version=5, plan_id="1",
            target=dict(target_id=self.spec["target_id"], capability_version=1,
                operator_spec=dict(id=self.spec["spec_id"], version=self.spec["version"], source_sha256=self.digest),
                world_size=1, device_counts=[0]), values=self.values,
            external_inputs=self.external, initialization=[], execution=self.steps,
            finalization=[], final_outputs=[self.bindings[v] for v in outputs])


def variants(plan, name, at, allow_boot=True):
    """Retain no-rescale and all legal one-Q rescale prefixes, with/without Boot."""
    seeds = [plan.clone()]
    if allow_boot:
        refreshed = plan.clone()
        try:
            refreshed.boot(name, at)
        except Infeasible:
            pass  # Explicitly reject this search edge; never repair a failed plan.
        else:
            seeds.append(refreshed)
    for seed in seeds:
        yield seed
        while True:
            reduced = seed.clone()
            try:
                reduced.rescale(name, at)
            except Infeasible:
                break
            yield reduced
            seed = reduced


def active_rows(graph):
    needed = set(graph["outputs"])
    rows = []
    for row in reversed(graph["ops"]):
        if row[0] in needed:
            rows.append(row)
            needed.update(row[2])
    return list(reversed(rows))


def search(graph, case, spec, digest, width=256):
    if width < 1:
        raise ValueError("beam width must be positive")
    rows = active_rows(graph)
    states = [Plan(spec, case, digest)]
    pruning = 0
    explored = 0
    for i, row in enumerate(rows):
        name, _, operands = row[:3]
        needed = set(graph["outputs"])
        for later in rows[i+1:]:
            needed.update(v for v in later[2] if v not in graph["constants"])
        merged = {}
        for plan in states:
            branches = [plan]
            for operand in dict.fromkeys(v for v in operands if v not in graph["constants"]):
                branches = [choice for branch in branches for choice in variants(branch, operand, name)]
            for branch in branches:
                try:
                    branch.apply(row, graph["constants"])
                except Infeasible:
                    continue
                for candidate in variants(branch, name, f"after:{name}", allow_boot=False):
                    explored += 1
                    candidate.bindings = {k: v for k, v in candidate.bindings.items() if k in needed}
                    # Exact float state and alias classes; no rounded state merging.
                    aliases = {}
                    key = []
                    for logical, value in sorted(candidate.bindings.items()):
                        aliases.setdefault(value, len(aliases))
                        a = candidate.state(value)
                        key.append((logical, a["level"], float(a["scale_log2"]).hex(), aliases[value]))
                    key = tuple(key)
                    rank = (candidate.boots, len(candidate.steps))
                    if key not in merged or rank < (merged[key].boots, len(merged[key].steps)):
                        merged[key] = candidate
        states = sorted(merged.values(), key=lambda p: (p.boots, len(p.steps)))
        pruning += max(0, len(states) - width)
        states = states[:width]
        if not states:
            raise Infeasible(f"reference search exhausted at {name}")
    finished = []
    for state in states:
        branches = [state]
        for name, bound in zip(graph["outputs"], case["outputs"]):
            branches = [v for p in branches for v in variants(p, name, "return")
                        if bound["min_scale_log2"] - 1e-8 <= v.state(v.bindings[name])["scale_log2"] <=
                        bound["max_scale_log2"] + 1e-8 and v.state(v.bindings[name])["level"] >= bound["min_level"]]
        finished.extend(branches)
    if not finished:
        raise Infeasible("reference search cannot meet output contract")
    best = min(finished, key=lambda p: (p.boots, len(p.steps)))
    result = best.export(graph["outputs"])
    stats = verify(result, spec, case, digest)
    zero = stats["boot_count"] == 0
    info = dict(origin="independent-python-search", beam_width=width,
        explored_transitions=explored, beam_discarded_states=pruning,
        boot_count=stats["boot_count"], global_optimality_proven=zero,
        proof="Boot counts are nonnegative; this legal plan uses zero." if zero else None,
        search_scope="Original live instruction order; coefficient scale 40; optional operand Boot and one-Q rescale prefixes; explicit Boot-contract alignment; upward unit alignment; shared future aliases. No global optimum claim.",
        decisions=best.events)
    return result, info


def materialize_recipe(graph, case, spec, digest, recipe):
    """Execute explicit human Boot choices; never insert a repair Boot."""
    if set(recipe) != {"boots_before", "rescale_after_each_operation", "explanation"}:
        raise ValueError("unexpected recipe fields")
    rows = active_rows(graph)
    if set(recipe["boots_before"]) - {row[0] for row in rows}:
        raise ValueError("recipe references an unknown instruction")
    plan = Plan(spec, case, digest)
    for row in rows:
        for name in recipe["boots_before"].get(row[0], []):
            plan.boot(name, row[0])
        plan.apply(row, graph["constants"])
        if recipe["rescale_after_each_operation"]:
            while True:
                a = plan.state(plan.bindings[row[0]])
                if a["level"] <= spec["levels"]["lower_bound"] or a["scale_log2"] - math.log2(
                    spec["context"]["rns_moduli"][a["level"]]) < case["min_ciphertext_scale_log2"]:
                    break
                plan.rescale(row[0], f"after:{row[0]}")
    result = plan.export(graph["outputs"])
    stats = verify(result, spec, case, digest)
    info = dict(origin="hand-authored-recipe", boot_count=stats["boot_count"],
        global_optimality_proven=stats["boot_count"] == 0,
        explanation=recipe["explanation"], recipe=recipe, decisions=plan.events)
    return result, info


def plan_bytes(plan):
    return (json.dumps(plan, indent=2, allow_nan=False) + "\n").encode()


def reference_metadata(plan, info, source):
    """Keep frozen expectations, without duplicating expanded plans/events."""
    result = {k: v for k, v in info.items() if k not in ("recipe", "decisions")}
    result.update(plan_source=source, plan_sha256=hashlib.sha256(plan_bytes(plan)).hexdigest())
    return result


def reference_plan(directory):
    """Rebuild a frozen reference and require its original byte-for-byte hash."""
    directory = Path(directory)
    metadata = read_json(directory / "reference.json")
    spec_path = directory.parent / "operator-spec.json"
    for key in ("graph", "case", "operator_spec", "recipe", "pattern"):
        if key in metadata["inputs_sha256"]:
            path = spec_path if key == "operator_spec" else directory / f"{key}.json"
            if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["inputs_sha256"][key]:
                raise ValueError(f"{directory.name}: frozen reference {key} hash mismatch")
    spec, case = read_json(spec_path), read_json(directory / "case.json")
    graph = read_json(directory / "graph.json")
    digest = "sha256:" + hashlib.sha256(spec_path.read_bytes()).hexdigest()
    if metadata["plan_source"] == "recipe":
        plan, _ = materialize_recipe(graph, case, spec, digest, read_json(directory / "recipe.json"))
    elif metadata["plan_source"] == "search":
        plan, _ = search(graph, case, spec, digest, metadata["beam_width"])
    else:
        raise ValueError(f"unknown reference source: {metadata['plan_source']}")
    if hashlib.sha256(plan_bytes(plan)).hexdigest() != metadata["plan_sha256"]:
        raise ValueError(f"{directory.name}: regenerated reference plan hash mismatch")
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-dir", type=Path, required=True)
    p.add_argument("--operator-spec", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--beam-width", type=int, default=256)
    p.add_argument("--recipe", type=Path, help="Explicit human decisions; do not run search")
    args = p.parse_args()
    graph = read_json(args.case_dir / "graph.json")
    case, spec = read_json(args.case_dir / "case.json"), read_json(args.operator_spec)
    digest = "sha256:" + hashlib.sha256(args.operator_spec.read_bytes()).hexdigest()
    if args.recipe:
        result, info = materialize_recipe(graph, case, spec, digest, read_json(args.recipe))
    else:
        result, info = search(graph, case, spec, digest, args.beam_width)
    info["inputs_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in (("graph", args.case_dir / "graph.json"), ("case", args.case_dir / "case.json"),
                           ("operator_spec", args.operator_spec), ("generator", Path(__file__)))}
    if args.recipe:
        info["inputs_sha256"]["recipe"] = hashlib.sha256(args.recipe.read_bytes()).hexdigest()
    info = reference_metadata(result, info, "recipe" if args.recipe else "search")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("manual", result), ("reference", info)):
        (args.output_dir / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
    print(case["case_id"], stats_summary(result, info), flush=True)


def stats_summary(result, info):
    return f"boots={info['boot_count']} steps={len(result['execution'])} origin={info['origin']}"


if __name__ == "__main__":
    main()
