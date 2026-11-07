#!/usr/bin/env python3
"""Compare native-scale RuntimePlan V5 schedules without a GPU or FHE library.

This is an algorithm-test verifier, not a replacement for the RuntimePlan
reader or encrypted accuracy tests. See README.md for the supported subset.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys


SCALE_TOLERANCE = 1e-8
HOST = {"kind": "host", "rank": 0}
OPS = {"add_cc", "add_cp", "sub_cc", "sub_cp", "mul_cc", "mul_cp",
       "negate", "rotate", "rescale", "mod_switch", "relinearize", "boot"}


class InvalidPlan(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidPlan(message)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(a, b):
    return number(a) and number(b) and abs(a - b) <= SCALE_TOLERANCE


def decoded_unit(scale_log2):
    """Decoded full-slot +1 in the scalar coefficient rounding model.

    Match positive C++ std::round, including half ties. This is not a
    per-slot quantizer for arbitrary CKKS plaintext vectors.
    """
    require(number(scale_log2) and 0 <= scale_log2 <= 1023,
            "unit encoding scale outside supported range")
    scale = math.exp2(scale_log2)
    coefficient = math.floor(scale)
    if scale - coefficient >= 0.5:
        coefficient += 1
    return coefficient / scale


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    def reject_constant(value):
        raise InvalidPlan(f"non-finite JSON constant: {value}")
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_object,
                      parse_constant=reject_constant)


def validate_case(case, spec):
    require(case["format_version"] == 1, "case format_version must be 1")
    require(isinstance(case["case_id"], str) and case["case_id"], "case_id is required")
    require(spec["spec_format_version"] in (1, 2), "unsupported OperatorSpec version")
    require(spec["rescale_mode"] == "native", "only native rescale is supported")
    context = spec["context"]
    degree = context["poly_degree"]
    require(type(degree) is int and degree >= 2 and degree & (degree - 1) == 0,
            "poly_degree must be a power of two")
    moduli = context["rns_moduli"]
    require(moduli and all(type(q) is int and q > 1 for q in moduli),
            "rns_moduli must contain positive integers greater than one")
    require([q.bit_length() for q in moduli] == context["rns_moduli_log2"],
            "rns_moduli_log2 disagrees with actual moduli")
    lo, hi = spec["levels"]["lower_bound"], spec["levels"]["upper_bound"]
    require(type(lo) is int and type(hi) is int and 0 <= lo <= hi < len(moduli),
            "invalid level bounds")
    ids = [p["profile_id"] for p in spec["boot_profiles"]]
    require(len(ids) == len(set(ids)), "duplicate Boot profile")
    for field in ("min_ciphertext_scale_log2", "min_plaintext_scale_log2",
                  "min_unit_plaintext_scale_log2"):
        require(number(case[field]) and case[field] >= 0, f"invalid {field}")
    require(case["inputs"] and case["outputs"], "case needs inputs and outputs")
    for bound in case["outputs"]:
        require(type(bound["min_level"]) is int and lo <= bound["min_level"] <= hi,
                "invalid output min_level")
        require(number(bound["min_scale_log2"]) and number(bound["max_scale_log2"])
                and 0 <= bound["min_scale_log2"] <= bound["max_scale_log2"],
                "invalid output scale interval")
    tolerance = case["numeric_tolerance"]
    require(all(number(tolerance[k]) and tolerance[k] >= 0 for k in ("abs", "rel")),
            "invalid numeric tolerance")
    if "unit_encoding" in case:
        policy = case["unit_encoding"]
        require(set(policy) == {"max_abs_error", "output_tolerance"},
                "invalid unit_encoding policy fields")
        require(number(policy["max_abs_error"]) and policy["max_abs_error"] >= 0,
                "invalid unit encoding error budget")
        require(set(policy["output_tolerance"]) == {"abs", "rel"} and
                all(number(x) and x >= 0 for x in policy["output_tolerance"].values()),
                "invalid unit-quantized output tolerance")
    require(case["test_vectors"], "at least one independent test vector is required")
    slots = degree // 2
    for vector in case["test_vectors"]:
        for field in ("inputs", "outputs"):
            require(len(vector[field]) == len(case[field]), f"test vector {field} arity")
            require(all(len(v) == slots and all(number(x) for x in v)
                        for v in vector[field]), f"test vector {field} must use all {slots} slots")


def verify(plan, spec, case, spec_sha256):
    """Return stats or raise InvalidPlan. Inputs/outputs bind by list position."""
    validate_case(case, spec)
    require(plan["format_version"] == 5, "only RuntimePlan V5 is supported")
    require("plaintext_bundle" not in plan and "waterline" not in plan,
            "export inline constants without plaintext scheduling")
    target = plan["target"]
    require(target["target_id"] == spec["target_id"], "target mismatch")
    require(target["operator_spec"] == {
        "id": spec["spec_id"], "version": spec["version"],
        "source_sha256": spec_sha256}, "OperatorSpec identity/hash mismatch")
    require(target["world_size"] == 1, "export without distributed placement")
    context = spec["context"]
    slots = context["poly_degree"] // 2
    moduli = context["rns_moduli"]
    profiles = {p["profile_id"]: p for p in spec["boot_profiles"]}
    desc = {}
    for value in plan["values"]:
        name = value["id"]
        require(isinstance(name, str) and name and name not in desc,
                f"duplicate or invalid value id: {name}")
        require(value["place"] == HOST, f"{name}: export without device placement")
        require(value["context"] == context["context_id"], f"{name}: context mismatch")
        level, scale, components = value["level"], value["scale_log2"], value["components"]
        require(type(level) is int and spec["levels"]["lower_bound"] <= level <=
                spec["levels"]["upper_bound"], f"{name}: level outside bounds")
        require(number(scale) and 0 <= scale < sum(math.log2(q) for q in moduli[:level+1]),
                f"{name}: scale exceeds modulus capacity or is invalid")
        require(type(value["ntt"]) is bool, f"{name}: invalid NTT state")
        require(value["kind"] in ("ciphertext", "plaintext"), f"{name}: invalid value kind")
        require(type(components) is int and
                (components == 1 if value["kind"] == "plaintext" else components >= 2),
                f"{name}: invalid component count")
        if value["kind"] == "ciphertext":
            require(scale + SCALE_TOLERANCE >= case["min_ciphertext_scale_log2"],
                    f"{name}: ciphertext scale below floor")
        desc[name] = value

    external, outputs = plan["external_inputs"], plan["final_outputs"]
    require(len(external) == len(case["inputs"]) and len(set(external)) == len(external),
            "external input arity or uniqueness mismatch")
    require(len(outputs) == len(case["outputs"]), "output arity mismatch")
    defined = set(external)
    for name, expected in zip(external, case["inputs"]):
        require(name in desc, f"missing input descriptor: {name}")
        for key in ("kind", "level", "scale_log2", "components", "ntt"):
            require(close(desc[name][key], expected[key]) if key == "scale_log2"
                    else desc[name][key] == expected[key], f"{name}: input {key} mismatch")
        require(desc[name]["kind"] == "ciphertext", "only ciphertext external inputs are supported")

    samples = [dict(zip(external, (list(x) for x in v["inputs"])))
               for v in case["test_vectors"]]
    unit_policy = case.get("unit_encoding")
    unit_samples = [dict(zip(external, (list(x) for x in v["inputs"])))
                    for v in case["test_vectors"]] if unit_policy is not None else []
    replays = [("plaintext", samples), ("unit-quantized", unit_samples)]
    unit_encodings = []
    counts = Counter({op: 0 for op in sorted(OPS)})
    rescale_levels = modswitch_levels = 0
    ordinal = 0
    encodes = 0
    for phase in ("initialization", "execution", "finalization"):
        for step in plan[phase]:
            location = f"{phase}[{ordinal}]"
            require(type(step["ordinal"]) is int and step["ordinal"] == ordinal,
                    f"{location}: ordinals must be globally consecutive")
            ordinal += 1
            kind = step["kind"]
            require(kind in ("encode", "compute"),
                    f"{location}: unsupported step {kind}; disable placement/memory scheduling")
            out = step["output"]
            require(out in desc and out not in defined, f"{location}: missing/redefined output {out}")
            after = desc[out]
            if kind == "encode":
                require(phase != "finalization" and after["kind"] == "plaintext",
                        f"{location}: invalid Encode output/phase")
                payload = step["payload"]
                require(payload["kind"] == "inline", f"{location}: only inline constants are supported")
                values = payload["values"]
                require(0 < len(values) <= slots and all(number(x) for x in values),
                        f"{location}: invalid inline payload")
                values = list(values) + [0.] * (slots - len(values))
                unit = all(x == 1 for x in values) or all(x == -1 for x in values)
                decoded = values
                if unit and unit_policy is not None:
                    factor = decoded_unit(after["scale_log2"])
                    error = abs(factor - 1)
                    unit_encodings.append({"value": out, "scale_log2": after["scale_log2"],
                                           "decoded_value": values[0] * factor, "abs_error": error})
                    require(error <= unit_policy["max_abs_error"],
                            f"{location}: unit encoding error {error} exceeds budget "
                            f"{unit_policy['max_abs_error']} (value {out})")
                    decoded = [x * factor for x in values]
                else:
                    floor = case["min_unit_plaintext_scale_log2" if unit else "min_plaintext_scale_log2"]
                    require(after["scale_log2"] + SCALE_TOLERANCE >= floor,
                            f"{location}: plaintext encoding scale {after['scale_log2']} below floor {floor}")
                for sample in samples:
                    sample[out] = values[:]
                for sample in unit_samples:
                    sample[out] = decoded[:]
                encodes += 1
                defined.add(out)
                continue

            op, inputs = step["op"], step["inputs"]
            require(op in OPS, f"{location}: unsupported operation {op}")
            require(spec["operators"][op]["supported"] is True, f"{location}: operator disabled: {op}")
            binary = op.startswith(("add_", "sub_", "mul_"))
            require(len(inputs) == (2 if binary else 1), f"{location}: operand arity")
            require(all(v in defined for v in inputs), f"{location}: use before definition")
            before = [desc[v] for v in inputs]
            a = before[0]
            require(a["kind"] == "ciphertext" and after["kind"] == "ciphertext",
                    f"{location}: operation requires ciphertext")
            require(step["place"] == HOST, f"{location}: unsupported compute place")
            require(all(v["ntt"] == a["ntt"] for v in before + [after]),
                    f"{location}: NTT state changed")
            attrs = step.get("attrs", {})
            required_attrs = {
                "rotate": {"steps"}, "rescale": {"target_level", "target_scale_log2"},
                "mod_switch": {"target_level"},
                "boot": {"target_level", "target_scale_log2", "target_components",
                         "operator_profile", "implementation"},
            }.get(op, set())
            require(set(attrs) == required_attrs, f"{location}: invalid attributes for {op}")
            level, scale, components = a["level"], a["scale_log2"], a["components"]
            if binary:
                b = before[1]
                require(b["kind"] == ("plaintext" if op.endswith("_cp") else "ciphertext"),
                        f"{location}: operand kind mismatch")
                require(b["level"] == level, f"{location}: operand level mismatch")
                if op.startswith("mul"):
                    scale += b["scale_log2"]
                    components += b["components"] - 1
                else:
                    require(close(scale, b["scale_log2"]), f"{location}: add/sub scale mismatch")
                    require(op.endswith("_cp") or b["components"] == components,
                            f"{location}: add/sub component mismatch")
            elif op in ("rescale", "mod_switch"):
                target_level = attrs["target_level"]
                require(type(target_level) is int and 0 <= target_level < level,
                        f"{location}: level must decrease")
                drop = level - target_level
                if op == "rescale":
                    require(drop <= spec["operators"][op]["max_levels_per_op"],
                            f"{location}: too many rescale levels")
                    scale -= sum(math.log2(q) for q in moduli[target_level+1:level+1])
                    require(close(scale, attrs["target_scale_log2"]),
                            f"{location}: rescale disagrees with actual moduli")
                    rescale_levels += drop
                else:
                    modswitch_levels += drop
                level = target_level
            elif op == "relinearize":
                require(components == 3, f"{location}: relinearize requires 3 components")
                components = 2
            elif op == "rotate":
                require(components == 2 and type(attrs["steps"]) is int
                        and attrs["steps"] % slots != 0, f"{location}: invalid rotation")
            elif op == "boot":
                profile_id = attrs["operator_profile"]
                require(profile_id in profiles, f"{location}: unknown Boot profile")
                p = profiles[profile_id]
                require(p["input_level_min"] <= level <= p["input_level_max"]
                        and components == p["input_components"], f"{location}: Boot input contract")
                if "input_scale_log2" in p:
                    require(close(scale, p["input_scale_log2"]), f"{location}: Boot input scale")
                require(attrs["implementation"] == p["implementation"],
                        f"{location}: Boot implementation mismatch")
                level, scale, components = p["output_level"], p["output_scale_log2"], p["output_components"]
                require(attrs["target_level"] == level and close(attrs["target_scale_log2"], scale)
                        and attrs["target_components"] == components, f"{location}: Boot output contract")
            require(after["level"] == level and close(after["scale_log2"], scale)
                    and after["components"] == components, f"{location}: output state disagrees with {op}")
            for replay_name, sample in ((name, sample) for name, batch in replays for sample in batch):
                left = sample[inputs[0]]
                if binary:
                    right = sample[inputs[1]]
                    if op.startswith("mul"):
                        result = [x * y for x, y in zip(left, right)]
                    elif op.startswith("add"):
                        result = [x + y for x, y in zip(left, right)]
                    else:
                        result = [x - y for x, y in zip(left, right)]
                elif op == "negate":
                    result = [-x for x in left]
                elif op == "rotate":
                    shift = attrs["steps"] % slots
                    result = left[shift:] + left[:shift]
                else:
                    result = left[:]
                require(all(number(x) for x in result), f"{location}: non-finite {replay_name} replay")
                sample[out] = result
            defined.add(out)
            counts[op] += 1

    require(defined == set(desc), "unused value descriptors")
    require(all(v in defined for v in outputs), "undefined final output")
    for name, bound in zip(outputs, case["outputs"]):
        value = desc[name]
        require(all(value[k] == bound[k] for k in ("kind", "components", "ntt")),
                f"{name}: output kind/components/NTT contract")
        require(value["level"] >= bound["min_level"] and
                bound["min_scale_log2"] - SCALE_TOLERANCE <= value["scale_log2"] <=
                bound["max_scale_log2"] + SCALE_TOLERANCE, f"{name}: output level/scale contract")
    errors = {}
    for replay_name, batch in replays:
        if not batch:
            continue
        maximum_error = 0.
        tolerance = (case["numeric_tolerance"] if replay_name == "plaintext"
                     else unit_policy["output_tolerance"])
        for index, (sample, vector) in enumerate(zip(batch, case["test_vectors"])):
            for name, expected in zip(outputs, vector["outputs"]):
                for slot, (actual, wanted) in enumerate(zip(sample[name], expected)):
                    maximum_error = max(maximum_error, abs(actual - wanted))
                    require(math.isclose(actual, wanted, abs_tol=tolerance["abs"], rel_tol=tolerance["rel"]),
                            f"{replay_name} test vector {index}, output {name}, slot {slot}: {actual} != {wanted}")
        errors[replay_name] = maximum_error
    return {"boot_count": counts["boot"], "operation_counts": dict(counts),
            "compute_count": sum(counts.values()), "encode_count": encodes,
            "rescale_levels": rescale_levels, "modswitch_levels": modswitch_levels,
            "value_count": len(desc), "test_vector_count": len(samples),
            "plaintext_max_abs_error": errors["plaintext"],
            "unit_quantization_checked": unit_policy is not None,
            "unit_quantized_max_abs_error": errors.get("unit-quantized"),
            "unit_encodings": unit_encodings,
            "output_states": [desc[v] for v in outputs]}


def compare(manual, automatic, spec, case, spec_sha256):
    report = {"case_id": case["case_id"], "plans": {}, "boot_delta": None,
              "comparison": "invalid", "semantic_check": (
                  "plaintext-and-unit-quantized-test-vectors" if "unit_encoding" in case
                  else "plaintext-test-vectors"),
              "optimality_proven": False, "encrypted_accuracy_checked": False}
    for name, plan in (("manual", manual), ("automatic", automatic)):
        try:
            stats = verify(plan, spec, case, spec_sha256)
            report["plans"][name] = {"valid": True, "stats": stats}
        except (InvalidPlan, KeyError, TypeError, ValueError, IndexError) as error:
            report["plans"][name] = {"valid": False, "error": f"{type(error).__name__}: {error}"}
    if all(p["valid"] for p in report["plans"].values()):
        delta = (report["plans"]["automatic"]["stats"]["boot_count"] -
                 report["plans"]["manual"]["stats"]["boot_count"])
        report["boot_delta"] = delta
        report["comparison"] = "better" if delta < 0 else "equal" if delta == 0 else "worse"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("case", "operator-spec", "manual", "automatic"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--output", type=Path, help="JSON report; also printed to stdout")
    parser.add_argument("--fail-on-worse", action="store_true")
    args = parser.parse_args()
    try:
        case, spec = read_json(args.case), read_json(args.operator_spec)
        validate_case(case, spec)
        digest = "sha256:" + hashlib.sha256(args.operator_spec.read_bytes()).hexdigest()
        report = compare(read_json(args.manual), read_json(args.automatic), spec, case, digest)
        report["sources"] = {name: {"path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in (("case", args.case), ("operator_spec", args.operator_spec),
                               ("manual", args.manual), ("automatic", args.automatic))}
    except (InvalidPlan, OSError, KeyError, TypeError, ValueError, IndexError) as error:
        report = {"comparison": "invalid", "error": f"{type(error).__name__}: {error}"}
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    sys.stdout.write(text)
    return 1 if report["comparison"] == "invalid" else (
        2 if args.fail_on_worse and report["comparison"] == "worse" else 0)


if __name__ == "__main__":
    sys.exit(main())
