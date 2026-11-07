"""Fixed-boundary budget checks for scale-alignment rewrites.

Bindings are supplied by the test author (or later by the pass's stable
semantic IDs), never inferred from SSA spelling. This checks state/Boot
preservation, not general graph equivalence. Validate the candidate with
verify.py as well. The before-plan may contain an inaccurate unit encoding.
"""
from verify import close, require


def steps(plan):
    return [step for phase in ("initialization", "execution", "finalization")
            for step in plan[phase]]


def operation_counts(plan):
    counts = {"encode": 0, "mul_cp": 0, "rescale": 0, "mod_switch": 0, "boot": 0}
    for step in steps(plan):
        name = "encode" if step["kind"] == "encode" else step["op"]
        counts[name] = counts.get(name, 0) + 1
    return counts


def verify_alignment_budget(before, after, contract):
    """Require identical boundary states and identical bound Boot operations.

    Extra Rescale instructions are allowed; extra boundary level loss is not.
    All external inputs, final outputs, Boot endpoints, and explicitly listed
    region exits must be bound. Internal/shared exits must be enumerated by
    the caller; this function cannot discover semantic correspondence.
    """
    require(contract["format_version"] == 1, "alignment budget: contract version")
    bindings = contract["boundaries"]
    require(bindings, "alignment budget: missing boundaries")
    for key in ("name", "before", "after"):
        require(all(isinstance(b[key], str) and b[key] for b in bindings) and
                len({b[key] for b in bindings}) == len(bindings),
                f"alignment budget: duplicate/invalid boundary {key}")
    mapping = {b["before"]: b["after"] for b in bindings}
    old, new = ({v["id"]: v for v in p["values"]} for p in (before, after))
    require(len(old) == len(before["values"]) and len(new) == len(after["values"]),
            "alignment budget: duplicate value descriptor")
    for field in ("external_inputs", "final_outputs"):
        require(all(v in mapping for v in before[field]),
                f"alignment budget: unbound {field}")
        require([mapping[v] for v in before[field]] == after[field],
                f"alignment budget: changed {field}")
    states = []
    for binding in bindings:
        require(binding["before"] in old and binding["after"] in new,
                f"alignment budget: missing boundary {binding['name']}")
        a, b = old[binding["before"]], new[binding["after"]]
        for field in ("level", "scale_log2", "kind", "components", "ntt", "context", "place"):
            same = close(a[field], b[field]) if field == "scale_log2" else a[field] == b[field]
            require(same, f"alignment budget: boundary {binding['name']} {field} changed "
                          f"{a[field]} -> {b[field]}")
        states.append({"name": binding["name"], "level": b["level"],
                       "scale_log2": b["scale_log2"], "level_delta": 0})
    boots = [[s for s in steps(p) if s.get("op") == "boot"] for p in (before, after)]
    require(len(boots[0]) == len(boots[1]), "alignment budget: Boot count changed")
    for a, b in zip(*boots):
        endpoints = a["inputs"] + [a["output"]]
        require(all(v in mapping for v in endpoints), "alignment budget: unbound Boot endpoint")
        require([mapping[v] for v in endpoints] == b["inputs"] + [b["output"]],
                "alignment budget: Boot binding changed")
        require(a["attrs"] == b["attrs"] and a["place"] == b["place"],
                "alignment budget: Boot contract changed")
    old_counts, new_counts = operation_counts(before), operation_counts(after)
    return {"boundaries": states, "boot_count": len(boots[1]),
            "operation_delta": {op: new_counts.get(op, 0) - old_counts.get(op, 0)
                                for op in sorted(old_counts.keys() | new_counts.keys())}}
