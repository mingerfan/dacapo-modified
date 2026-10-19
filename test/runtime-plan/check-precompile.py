#!/usr/bin/env python3
"""Manual CKKS checkpoints must preserve full-pipeline plans and payloads."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def source(depth):
    cipher = "tensor<1x!earth.ci<0 * 0>>"
    plain = "tensor<1x!earth.pl<0 * 0>>"
    rows = [
        f"module {{ func.func @probe(%x: {cipher}) -> {cipher} {{",
        '%p = "earth.constant"() {value = dense<[1.01, -0.1, '
        '1.23456789012345, 0.0, 3.7]> : tensor<5xf64>, '
        f'rms_var = 1.01 : f64}} : () -> {plain}',
    ]
    previous = "%x"
    for index in range(depth):
        rows.append(f'%m{index} = "earth.mul"({previous}, %p) : '
                    f'({cipher}, {plain}) -> {cipher}')
        previous = f"%m{index}"
    rows += [
        f'%a = "earth.rotate"({previous}) <{{offset = array<i64: 1>}}> : '
        f'({cipher}) -> {cipher}',
        f'%b = "earth.rotate"({previous}) <{{offset = array<i64: 2>}}> : '
        f'({cipher}) -> {cipher}',
        f'%sum = "earth.add"(%a, %b) : ({cipher}, {cipher}) -> {cipher}',
        f'%y = "earth.add"(%sum, %x) : ({cipher}, {cipher}) -> {cipher}',
        f"return %y : {cipher}", "}}",
    ]
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hecate-opt", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    fixtures = args.source_dir / "test/runtime-plan"
    with tempfile.TemporaryDirectory(prefix="dacapo-precompile-") as directory:
        root = Path(directory)
        traced = root / "source.mlir"
        profile = json.loads((args.source_dir / "config.json").read_text())
        profile.update(polynomialDegree=16, rescalingFactor=120,
                       bootstrapLevelLowerBound=1, bootstrapLevelUpperBound=4,
                       levelLowerBound=1, levelUpperBound=4)
        config = root / "config.json"
        config.write_text(json.dumps(profile))
        common = [f"--ckks-config={config}", "--mlir-disable-threading",
                  "--mlir-timing"]
        export = [
            "--runtime-plan-id=1", "--runtime-plan-target-id=test",
            "--runtime-plan-context-id=test", "--runtime-plan-operator-spec-id=test",
            "--runtime-plan-operator-spec-sha256=sha256:" + "0" * 64,
            "--runtime-plan-boot-profile=test-boot",
            "--runtime-plan-boot-implementation=decrypt_reencrypt",
            "--runtime-plan-inline-payload-max-bytes=0",
        ]
        spec = json.loads((fixtures / "placement-operator-spec.json").read_text())
        # Expand fixture costs to cover the physical levels of a lazy target.
        spec["levels"] = {"lower_bound": 4, "upper_bound": 49}
        spec["context"].update(rns_moduli_log2=[40] * 50,
                               max_modulus_log2=40, default_scale_log2=120)
        for operator in spec["operators"].values():
            if "latency_us_by_level" in operator:
                operator["latency_us_by_level"] = [operator["latency_us_by_level"][-1]] * 50
        spec["operators"]["rescale"]["max_levels_per_op"] = 4
        boot = spec["boot_profiles"][0]
        boot.update(input_level_min=1, input_level_max=49, output_level=49,
                    output_scale_log2=120, latency_us_by_input_level=[500] * 50)
        spec["rescale_mode"] = "lazy"
        operator_spec = root / "operator-spec.json"
        operator_spec.write_text(json.dumps(spec))
        placement = [
            "--runtime-plan-device-counts=4",
            f"--runtime-plan-operator-spec-path={operator_spec}",
            "--runtime-plan-intra-rank-communication-cost=1",
        ]
        modes = {
            "host": ["--runtime-plan-format=json"],
            "gpu-memory": ["--runtime-plan-format=json", *placement,
                           "--runtime-plan-memory", "--runtime-plan-memory-report"],
            "lazy-waterline": [
                "--runtime-plan-format=binary", *placement,
                f"--runtime-plan-physical-level-operator-spec-path={operator_spec}",
                "--runtime-plan-levels-per-logical-level=4",
                "--runtime-plan-plaintext-schedule=waterline",
                "--runtime-plan-gpu-budget-bytes=1048576",
                "--runtime-plan-host-budget-bytes=4194304",
                "--runtime-plan-pinned-budget-bytes=1048576",
                "--runtime-plan-raw-budget-bytes=1048576",
            ],
        }

        def run(pipeline, path, output, options):
            output.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                [str(args.hecate_opt), pipeline, *common, *options,
                 str(path), "-o", str(output)],
                cwd=args.source_dir, capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, result.stderr
            return result

        def artifacts(directory):
            return {str(path.relative_to(directory)): path.read_bytes()
                    for path in directory.rglob("*")
                    if path.is_file() and path.name != "result.mlir"}

        for strategy in ("optimized", "greedy", "depth-dp"):
            # Cover optimized lowering and multiple fast-planner Boot boundaries.
            traced.write_text(source(1 if strategy == "optimized" else 12))
            early = [f"--boot-placement={strategy}", "--waterline=40"]
            caches = []
            for extension in ("mlir", "mlirbc"):
                cache = root / strategy / extension / f"checkpoint.{extension}"
                options = early + (["--emit-bytecode",
                                    "--mlir-elide-elementsattrs-if-larger=1"]
                                   if extension == "mlirbc" else [])
                result = run("--dacapo-precompile", traced, cache, options)
                assert "AssignPlacement" not in result.stderr
                assert "EmitRuntimePlan" not in result.stderr
                assert list(cache.parent.iterdir()) == [cache]
                if extension == "mlirbc":
                    assert cache.read_bytes()[:4] == b"ML\xefR"
                else:
                    text = cache.read_text()
                    if strategy != "optimized":
                        assert "ckks.bootstrapc" in text
                    assert '"earth.' not in text
                    assert "dist." not in text
                caches.append(cache)

            for mode, late in modes.items():
                full = root / strategy / mode / "full" / "result.mlir"
                run("--dacapo", traced, full, [*early, *export, *late])
                expected = artifacts(full.parent)
                assert any(".runtime-plan." in name for name in expected)
                assert any(name.endswith("data.bin") for name in expected)
                for cache in caches:
                    before = cache.read_bytes()
                    resumed = (root / strategy / mode / cache.suffix[1:]
                               / "result.mlir")
                    result = run("--ckks-to-runtime-plan", cache, resumed,
                                 [*export, *late])
                    assert cache.read_bytes() == before
                    for pass_name in ("DaCapoPlanner", "GreedyBootstrapPlacement",
                                      "EarthToCKKSConversion", "RemoveBootstrap"):
                        assert pass_name not in result.stderr, result.stderr
                    assert "EmitRuntimePlan" in result.stderr
                    if mode != "host":
                        assert "AssignPlacement" in result.stderr
                        assert "MaterializeCommunication" in result.stderr
                    assert artifacts(resumed.parent) == expected, (strategy, mode, cache)
    print("Manual text/bytecode checkpoints match full Boot, placement, memory and Waterline pipelines")


if __name__ == "__main__":
    main()
