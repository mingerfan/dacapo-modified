#!/usr/bin/env python3

import argparse
import hashlib
import json
import struct
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


def load_bundle(plan: dict, directory: Path) -> tuple[dict, dict[str, bytes]]:
    manifest_path = directory / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    assert plan["plaintext_bundle"]["manifest_sha256"] == (
        "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
    )
    manifest = json.loads(manifest_bytes)
    assert manifest["bundle_id"] == plan["plaintext_bundle"]["id"]
    assert manifest["version"] == plan["plaintext_bundle"]["version"]
    blobs = {}
    for entry in manifest["blobs"]:
        content = entry["content"]
        data = (directory / "data" / f"{content[7:]}.bin").read_bytes()
        assert len(data) == entry["byte_length"]
        assert content == "sha256:" + hashlib.sha256(data).hexdigest()
        blobs[content] = data
    return manifest, blobs


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
        upscale_payload = upscale["initialization"][0]["payload"]
        assert upscale_payload["kind"] == "bundle"
        expected_slots = json.loads(
            (args.source_dir / "config.json").read_text(encoding="utf-8")
        )["polynomialDegree"] // 2
        upscale_manifest, upscale_blobs = load_bundle(
            upscale, temp / "upscale.upscale.bundle"
        )
        assert len(upscale_manifest["blobs"]) == 1
        upscale_data = upscale_blobs[upscale_payload["content"]]
        upscale_values = struct.unpack(f"<{expected_slots}d", upscale_data)
        assert set(upscale_values) == {1.0}
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
        assert "plaintext_bundle" not in constant
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

        zero_rotate, _ = run_pipeline(
            args.hecate_opt, args.source_dir, temp,
            "zero-rotate", "zero_rotate", 6,
            ["canonicalize", "convert-earth-to-ckks"],
        )
        assert len(zero_rotate["execution"]) == 1
        assert zero_rotate["execution"][0]["op"] == "rotate"
        assert zero_rotate["execution"][0]["attrs"] == {"steps": 1}
        assert zero_rotate["execution"][0]["inputs"] == ["0"]

        bundle_reuse, _ = run_pipeline(
            args.hecate_opt, args.source_dir, temp,
            "bundle-reuse", "bundle_reuse", 7,
            ["convert-earth-to-ckks"],
        )
        payloads = [item["payload"] for item in bundle_reuse["initialization"]]
        assert [payload["kind"] for payload in payloads] == ["bundle", "bundle"]
        assert payloads[0]["content"] == payloads[1]["content"]
        reuse_manifest, reuse_blobs = load_bundle(
            bundle_reuse, temp / "bundle-reuse.bundle_reuse.bundle"
        )
        assert len(reuse_manifest["blobs"]) == 1
        reuse_values = struct.unpack("<600d", reuse_blobs[payloads[0]["content"]])
        assert set(reuse_values) == {1.25}


if __name__ == "__main__":
    main()
