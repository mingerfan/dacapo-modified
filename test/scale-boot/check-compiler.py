#!/usr/bin/env python3
"""Verify real compiler exports against independently computed vector outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import tempfile

from verify import InvalidPlan, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hecate-opt", type=Path, required=True)
    args = parser.parse_args()
    compiler = Path(__file__).resolve().parents[2]
    runtime = compiler.parents[1]
    generate = runpy.run_path(str(runtime / "integrations/dacapo/generate_resnet20_manual_profiles.py"))["generate"]
    source = runpy.run_path(str(compiler / "test/runtime-plan/check-native-planning.py"))["source"]
    with tempfile.TemporaryDirectory(prefix="scale-boot-compiler-") as directory:
        out = Path(directory)
        config, spec = generate(out)
        # Small SIMD vectors exercise all slots; no encrypted execution occurs.
        config["polynomialDegree"] = spec["context"]["poly_degree"] = 16
        (out / "compiler.json").write_text(json.dumps(config))
        (out / "operator-spec.json").write_text(json.dumps(spec))
        digest = "sha256:" + hashlib.sha256((out / "operator-spec.json").read_bytes()).hexdigest()
        inputs = [[1, 0, .999999, 1.000001, -1, .999998, .5, -.5],
                  [0, -1, 1, .999997, .999996, -.999999, -.5, .5]]
        case = {
            "format_version": 1, "case_id": "compiler-deep-residual",
            "inputs": [{"kind": "ciphertext", "level": 30, "scale_log2": 40,
                        "components": 2, "ntt": True}],
            "outputs": [{"kind": "ciphertext", "min_level": 4, "min_scale_log2": 40,
                         "max_scale_log2": 500, "components": 2, "ntt": True}],
            "min_ciphertext_scale_log2": 40, "min_plaintext_scale_log2": 40,
            # Legacy threshold alignment permits low-scale encodings of one.
            # This is its structural contract, not an accuracy endorsement.
            "min_unit_plaintext_scale_log2": 0,
            "numeric_tolerance": {"abs": 1e-10, "rel": 1e-10}, "test_vectors": [],
        }
        for vector in inputs:
            expected = []
            for x in vector:
                y = x
                for _ in range(18):
                    y *= y
                expected.append(y + x)
            case["test_vectors"].append({"inputs": [vector], "outputs": [expected]})
        (out / "input.mlir").write_text(source(18, residual=True).replace("tensor<1x", "tensor<8x"))
        for strategy in ("greedy", "depth-dp", "optimized"):
            prefix = out / strategy
            command = [str(args.hecate_opt.resolve()), str(out / "input.mlir"), "--dacapo",
                f"--boot-placement={strategy}", "--native-scale-policy=threshold",
                f"--ckks-config={out}/compiler.json", "--waterline=40", "--bootstrap-scale=40",
                "--runtime-plan-format=json", "--runtime-plan-inline-payload-max-bytes=1048576",
                f"--runtime-plan-physical-level-operator-spec-path={out}/operator-spec.json",
                f"--runtime-plan-operator-spec-path={out}/operator-spec.json",
                f"--runtime-plan-boot-profile={spec['boot_profiles'][0]['profile_id']}",
                f"--runtime-plan-target-id={spec['target_id']}",
                f"--runtime-plan-context-id={spec['context']['context_id']}",
                f"--runtime-plan-operator-spec-id={spec['spec_id']}",
                f"--runtime-plan-operator-spec-version={spec['version']}",
                f"--runtime-plan-operator-spec-sha256={digest}", "-o", str(prefix) + ".mlir"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            if result.returncode:
                raise RuntimeError(result.stderr)
            plan = json.loads(Path(str(prefix) + ".squares.runtime-plan.json").read_text())
            stats = verify(plan, spec, case, digest)
            if not stats["boot_count"]:
                raise AssertionError("deep test must exercise Boot verification")
            print(strategy, json.dumps({k: stats[k] for k in
                ("boot_count", "rescale_levels", "test_vector_count", "plaintext_max_abs_error")}))
            # The same plan must be rejected if the benchmark requires a
            # stronger floor. Never silently relax the user's case contract.
            strict_case = dict(case, min_unit_plaintext_scale_log2=40)
            try:
                verify(plan, spec, strict_case, digest)
            except InvalidPlan as error:
                if "plaintext encoding scale" not in str(error):
                    raise
            else:
                raise AssertionError("expected low-scale alignment to fail the strict contract")


if __name__ == "__main__":
    main()
