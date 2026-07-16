#!/usr/bin/env python3

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


SHA256 = "sha256:" + "0" * 64


def run_pipeline(hecate_opt: Path, source_dir: Path, temp: Path,
                 fixture: str, function: str, plan_id: int,
                 conversions: list[str], boot: bool = False) -> tuple[dict, Path]:
    prefix = temp / fixture
    options = [
        f"prefix={prefix}",
        f"plan-id={plan_id}",
        "target-id=test-target",
        "capability-version=1",
        "operator-spec-id=test-spec",
        "operator-spec-version=1",
        f"operator-spec-sha256={SHA256}",
        "context-id=test-context",
        "device-count=0",
        "ntt=true",
    ]
    if boot:
        options.extend([
            "boot-profile=test-boot",
            "boot-implementation=native",
        ])
    passes = ",".join(conversions + [f"emit-runtime-plan{{{' '.join(options)}}}"])
    output_mlir = temp / f"{fixture}.ckks.mlir"
    subprocess.run(
        [
            str(hecate_opt),
            str(source_dir / "test/runtime-plan" / f"{fixture}.mlir"),
            f"-p=builtin.module(func.func({passes}))",
            "-o",
            str(output_mlir),
        ],
        cwd=source_dir,
        check=True,
    )
    plan_path = Path(f"{prefix}.{function}.runtime-plan.json")
    return json.loads(plan_path.read_text(encoding="utf-8")), output_mlir


def value(plan: dict, value_id: str) -> dict:
    return next(item for item in plan["values"] if item["id"] == value_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hecate-opt", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="dacapo-runtime-plan-") as directory:
        temp = Path(directory)

        mul, mul_mlir = run_pipeline(
            args.hecate_opt, args.source_dir, temp,
            "mul-relinearize", "mul_relinearize", 1,
            ["convert-earth-to-ckks"],
        )
        assert [op["op"] for op in mul["execution"]] == [
            "mul_cc", "relinearize"
        ]
        assert (value(mul, "2")["components"],
                value(mul, "2")["scale_log2"]) == (3, 40)
        assert (value(mul, "3")["components"],
                value(mul, "3")["scale_log2"]) == (2, 40)

        retired = subprocess.run(
            [
                str(args.hecate_opt), str(mul_mlir),
                f"-p=builtin.module(func.func(emit-hevm{{prefix={temp / 'hevm'}}}))",
                "-o", str(temp / "hevm.mlir"),
            ],
            cwd=args.source_dir,
            text=True,
            capture_output=True,
        )
        assert retired.returncode != 0
        assert "HEVM emission has been retired" in retired.stderr

        upscale, _ = run_pipeline(
            args.hecate_opt, args.source_dir, temp,
            "upscale", "upscale", 2,
            ["convert-earth-to-ckks", "convert-upscale-to-mulcp"],
        )
        assert upscale["execution"][0]["op"] == "mul_cp"
        assert upscale["initialization"][0]["payload"]["kind"] == "inline"
        expected_slots = json.loads(
            (args.source_dir / "config.json").read_text(encoding="utf-8")
        )["polynomialDegree"] // 2
        assert len(upscale["initialization"][0]["payload"]["values"]) == expected_slots
        assert set(upscale["initialization"][0]["payload"]["values"]) == {1.0}
        assert value(upscale, "1")["scale_log2"] == 20
        assert value(upscale, "2")["scale_log2"] == 40

        constant, _ = run_pipeline(
            args.hecate_opt, args.source_dir, temp,
            "inline-constant", "inline_constant", 3,
            ["convert-earth-to-ckks", "convert-upscale-to-mulcp"],
        )
        assert constant["initialization"][0]["payload"]["values"] == [
            1.0, 2.0, 3.0, 4.0
        ]
        assert constant["execution"][0]["op"] == "add_cp"

        bootstrap, _ = run_pipeline(
            args.hecate_opt, args.source_dir, temp,
            "bootstrap-levels", "bootstrap_levels", 4,
            ["convert-earth-to-ckks"], boot=True,
        )
        assert (value(bootstrap, "0")["level"],
                value(bootstrap, "1")["level"]) == (3, 16)
        assert bootstrap["execution"][0]["attrs"] == {
            "target_level": 16,
            "target_scale_log2": 40,
            "target_components": 2,
            "operator_profile": "test-boot",
            "implementation": "native",
        }

        compute, _ = run_pipeline(
            args.hecate_opt, args.source_dir, temp,
            "compute-ops", "compute_ops", 5, [],
        )
        assert [op["op"] for op in compute["execution"]] == [
            "add_cc", "negate", "rotate", "rescale", "mod_switch"
        ]
        assert compute["execution"][2]["attrs"] == {"steps": 1}
        assert compute["execution"][3]["attrs"] == {
            "target_level": 4,
            "target_scale_log2": 20,
        }
        assert compute["execution"][4]["attrs"] == {"target_level": 3}


if __name__ == "__main__":
    main()
