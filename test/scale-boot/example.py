#!/usr/bin/env python3
"""Write a small manual/automatic comparison fixture (no cryptographic claim)."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from verify import HOST, OPS


def make_example():
    moduli = [257, 769, 1153]
    scale = 20 - math.log2(moduli[-1])
    spec = {
        "spec_format_version": 1, "spec_id": "scale-boot-example", "version": 1,
        "status": "placeholder", "target_id": "scale-boot-example", "rescale_mode": "native",
        "context": {"context_id": "example", "poly_degree": 8,
                    "rns_moduli": moduli, "rns_moduli_log2": [q.bit_length() for q in moduli],
                    "max_modulus_log2": 11, "default_scale_log2": 10},
        "levels": {"lower_bound": 0, "upper_bound": 2},
        "operators": {op: {"supported": True, "latency_us_by_level": None,
                           "noise_by_level": None} for op in sorted(OPS - {"boot"})},
        "boot_profiles": [{"profile_id": "example-boot", "implementation": "native",
            "input_level_min": 1, "input_level_max": 1, "input_components": 2,
            "output_level": 2, "output_scale_log2": 10, "output_components": 2,
            "latency_us": 1, "host_requirements": {"needs_secret_key": False,
                                                     "needs_host_compute": False}}],
    }
    spec["operators"]["boot"] = {"supported": True}
    spec["operators"]["rescale"]["max_levels_per_op"] = 1
    spec_text = json.dumps(spec, indent=2) + "\n"
    digest = "sha256:" + hashlib.sha256(spec_text.encode()).hexdigest()

    def value(name, level, scale, components=2):
        return {"id": name, "kind": "ciphertext", "place": dict(HOST),
                "context": "example", "level": level, "scale_log2": scale,
                "components": components, "ntt": True}

    def compute(index, op, inputs, output, **attrs):
        step = {"ordinal": index, "kind": "compute", "op": op,
                "place": dict(HOST), "inputs": inputs, "output": output}
        if attrs:
            step["attrs"] = attrs
        return step

    manual = {
        "format_version": 5, "plan_id": "1",
        "target": {"target_id": spec["target_id"], "capability_version": 1,
            "operator_spec": {"id": spec["spec_id"], "version": 1, "source_sha256": digest},
            "world_size": 1, "device_counts": [0]},
        "values": [value("0", 2, 10), value("1", 2, 20, 3), value("2", 2, 20),
                   value("3", 1, scale), value("4", 1, scale)],
        "external_inputs": ["0"], "initialization": [],
        "execution": [compute(0, "mul_cc", ["0", "0"], "1"),
            compute(1, "relinearize", ["1"], "2"),
            compute(2, "rescale", ["2"], "3", target_level=1, target_scale_log2=scale),
            compute(3, "rotate", ["3"], "4", steps=1)],
        "finalization": [], "final_outputs": ["4"],
    }
    automatic = deepcopy(manual)
    automatic["plan_id"] = "2"
    automatic["values"].append(value("5", 2, 10))
    automatic["execution"].append(compute(4, "boot", ["4"], "5", target_level=2,
        target_scale_log2=10, target_components=2, operator_profile="example-boot", implementation="native"))
    automatic["final_outputs"] = ["5"]
    case = {
        "format_version": 1, "case_id": "square-rotate-example",
        "inputs": [{"kind": "ciphertext", "level": 2, "scale_log2": 10,
                    "components": 2, "ntt": True}],
        "outputs": [{"kind": "ciphertext", "min_level": 1, "min_scale_log2": 9,
                     "max_scale_log2": 10, "components": 2, "ntt": True}],
        "min_ciphertext_scale_log2": 9, "min_plaintext_scale_log2": 9,
        "min_unit_plaintext_scale_log2": 3,
        "numeric_tolerance": {"abs": 1e-12, "rel": 1e-12},
        "test_vectors": [
            {"inputs": [[.25, -.5, .75, 1]], "outputs": [[.25, .5625, 1, .0625]]},
            {"inputs": [[1, 2, -3, 4]], "outputs": [[4, 9, 16, 1]]},
        ],
    }
    return case, spec, spec_text, digest, manual, automatic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    case, _, spec_text, _, manual, automatic = make_example()
    (args.output_dir / "operator-spec.json").write_text(spec_text)
    for name, data in (("case", case), ("manual", manual), ("automatic", automatic)):
        (args.output_dir / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
