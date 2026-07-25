#!/usr/bin/env python3

import argparse
import hashlib
import json
import re
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
    assert "tensor.empty" not in output_mlir.read_text(encoding="utf-8")
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


def run_placement_pipeline(hecate_opt: Path, source_dir: Path, temp: Path,
                           device_counts: str, plan_id: int,
                           fixture: str = "placement-fanout",
                           function: str = "placement_fanout",
                           boot_profile: str | None = None,
                           communication_profile: Path | None = None,
                           ) -> tuple[dict, str]:
    spec_path = source_dir / "test/runtime-plan/placement-operator-spec.json"
    spec_digest = "sha256:" + hashlib.sha256(spec_path.read_bytes()).hexdigest()
    prefix = temp / f"{fixture}-{device_counts}"
    boot_option = f" boot-profile={boot_profile}" if boot_profile else ""
    communication_option = (
        f" communication-profile={communication_profile}"
        if communication_profile else ""
    )
    emit_boot_option = (
        f" boot-profile={boot_profile} boot-implementation=decrypt_reencrypt"
        if boot_profile else ""
    )
    placement = (
        "assign-ckks-placement{"
        f"device-counts={device_counts} operator-spec={spec_path} "
        "intra-rank-communication-cost=10 "
        f"inter-rank-communication-cost=20{boot_option}"
        f"{communication_option}}}"
    )
    emit = (
        "emit-runtime-plan{"
        f"prefix={prefix} plan-id={plan_id} "
        "target-id=dacapo-placement-test capability-version=1 "
        "operator-spec-id=dacapo-placement-test-v1 "
        f"operator-spec-version=1 operator-spec-sha256={spec_digest} "
        f"context-id=test-context device-count=0 ntt=true{emit_boot_option}}}"
    )
    output = temp / f"{fixture}-{device_counts}.mlir"
    subprocess.run(
        [
            str(hecate_opt),
            str(source_dir / f"test/runtime-plan/{fixture}.mlir"),
            f"-p=builtin.module(func.func({placement},"
            f"materialize-ckks-communication,{emit}))",
            "-o", str(output),
        ],
        cwd=source_dir,
        check=True,
    )
    plan_path = Path(f"{prefix}.{function}.runtime-plan.json")
    return json.loads(plan_path.read_text(encoding="utf-8")), output.read_text(
        encoding="utf-8"
    )


def place_key(place: dict) -> tuple[int, int]:
    return place["rank"], place.get("index", -1)


def rotate_decomposition_term_count(step: int, slot_count: int) -> int:
    normalized = step % slot_count
    if normalized > slot_count // 2:
        normalized -= slot_count
    assert normalized != 0
    return abs(normalized).bit_count()


def verify_rotate_costs(mlir: str) -> None:
    durations = {}
    for line in mlir.splitlines():
        if '"ckks.rotatec"' not in line or "dist.schedule_start" not in line:
            continue
        logical_id = int(re.search(
            r"dist.logical_id = ([0-9]+)", line
        ).group(1))
        start = int(re.search(
            r"dist.schedule_start = ([0-9]+)", line
        ).group(1))
        finish = int(re.search(
            r"dist.schedule_finish = ([0-9]+)", line
        ).group(1))
        durations[logical_id] = finish - start

    # poly_degree=16 gives 8 slots. Each Rotate costs 1000 us per nonzero
    # signed-binary term after normalizing the step to the slot count.
    assert durations == {
        1: 1000, 2: 1000, 3: 2000, 4: 1000,
        5: 2000, 6: 1000, 7: 1000,
        8: 1000, 9: 1000, 10: 2000, 11: 1000,
        12: 2000, 13: 1000, 14: 1000,
        15: 1000, 16: 1000,
    }


def verify_placement(plan: dict, mlir: str,
                     expected_device_counts: list[int]) -> None:
    assert plan["target"]["world_size"] == len(expected_device_counts)
    assert plan["target"]["device_counts"] == expected_device_counts
    descriptions = {item["id"]: item for item in plan["values"]}
    instructions = plan["initialization"] + plan["execution"]
    transfers = [item for item in instructions if item["kind"] == "transfer"]
    computes = [item for item in instructions if item["kind"] == "compute"]
    assert len(computes) == 31
    assert transfers
    assert all(item["hint"] == "point_to_point" for item in transfers)
    assert all(len(item["inputs"]) == len(item["outputs"]) == 1
               for item in transfers)
    assert all(int(item["outputs"][0]) > 31 for item in transfers)

    for phase in (plan["initialization"], plan["execution"]):
        producer_indices = {
            item["output"]: index
            for index, item in enumerate(phase)
            if item["kind"] in ("encode", "compute")
        }
        for index, item in enumerate(phase):
            if item["kind"] != "transfer":
                continue
            input_id = item["inputs"][0]
            producer_index = producer_indices.get(input_id)
            if producer_index is None:
                continue
            assert producer_index < index
            assert all(
                intervening["kind"] == "transfer"
                and intervening["inputs"] == [input_id]
                for intervening in phase[producer_index + 1:index]
            )

    expected_places = set()
    for rank, count in enumerate(expected_device_counts):
        if count == 0:
            expected_places.add((rank, -1))
        else:
            expected_places.update((rank, device) for device in range(count))
    compute_places = {place_key(item["place"]) for item in computes}
    assert compute_places == expected_places
    for item in computes:
        output_place = place_key(descriptions[item["output"]]["place"])
        assert output_place == place_key(item["place"])
        assert all(place_key(descriptions[value_id]["place"]) == output_place
                   for value_id in item["inputs"])

    schedule = {}
    intervals: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for line in mlir.splitlines():
        if '"ckks.' not in line or "dist.schedule_start" not in line:
            continue
        logical_id = int(re.search(r"dist.logical_id = ([0-9]+)", line).group(1))
        rank = int(re.search(r"dist.rank = ([0-9]+)", line).group(1))
        device = int(re.search(r"dist.device = (-?[0-9]+)", line).group(1))
        start = int(re.search(r"dist.schedule_start = ([0-9]+)", line).group(1))
        finish = int(re.search(r"dist.schedule_finish = ([0-9]+)", line).group(1))
        op = re.search(r'"ckks\.([a-z]+)"', line).group(1)
        if op == "rotatec":
            step = int(re.search(
                r"offset = array<i64: (-?[0-9]+)>", line
            ).group(1))
            expected_duration = (
                1000 * rotate_decomposition_term_count(step, 8)
            )
        else:
            expected_duration = 100
        assert finish - start == expected_duration
        schedule[str(logical_id)] = (start, finish, (rank, device))
        intervals.setdefault((rank, device), []).append((start, finish))
    assert set(schedule) == {str(value_id) for value_id in range(1, 32)}
    for placed_intervals in intervals.values():
        placed_intervals.sort()
        assert all(left[1] <= right[0]
                   for left, right in zip(placed_intervals,
                                          placed_intervals[1:]))

    transfer_by_output = {item["outputs"][0]: item for item in transfers}

    def ready_time(value_id: str) -> int:
        if value_id in schedule:
            return schedule[value_id][1]
        transfer = transfer_by_output.get(value_id)
        if transfer is None:
            assert value_id == "0"
            return 0
        source = place_key(transfer["sources"][0])
        destination = place_key(transfer["destinations"][0])
        cost = 10 if source[0] == destination[0] else 20
        return ready_time(transfer["inputs"][0]) + cost

    for item in computes:
        start, _, place = schedule[item["output"]]
        assert place == place_key(item["place"])
        assert start >= max(ready_time(value_id) for value_id in item["inputs"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hecate-opt", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="dacapo-runtime-plan-") as directory:
        temp = Path(directory)

        mul, _ = run_pipeline(
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

        invalid_rotate = subprocess.run(
            [
                str(args.hecate_opt),
                str(args.source_dir / "test/runtime-plan/invalid-rotate-plaintext.mlir"),
                "-o", str(temp / "invalid-rotate-plaintext.mlir"),
            ],
            cwd=args.source_dir,
            text=True,
            capture_output=True,
        )
        assert invalid_rotate.returncode != 0
        assert "source must be a two-component ciphertext" in invalid_rotate.stderr

        invalid_relinearize = subprocess.run(
            [
                str(args.hecate_opt),
                str(args.source_dir / "test/runtime-plan/invalid-relinearize.mlir"),
                "-o", str(temp / "invalid-relinearize.mlir"),
            ],
            cwd=args.source_dir,
            text=True,
            capture_output=True,
        )
        assert invalid_relinearize.returncode != 0
        assert "requires components 3 -> 2" in invalid_relinearize.stderr

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

        lazy_spec = (
            args.source_dir /
            "test/runtime-plan/lazy-physical-level-operator-spec.json"
        )
        physical_levels = (
            "materialize-ckks-physical-levels{"
            f"operator-spec={lazy_spec} levels-per-logical-level=4}}"
        )
        lazy_placement = (
            "assign-ckks-placement{"
            f"device-counts=1 operator-spec={lazy_spec} "
            "intra-rank-communication-cost=1000 "
            "inter-rank-communication-cost=10000}"
        )
        lazy, lazy_mlir_path = run_pipeline(
            args.hecate_opt, args.source_dir, temp,
            "lazy-physical-levels", "lazy_physical_levels", 8,
            [physical_levels, lazy_placement,
             "materialize-ckks-communication"],
        )
        assert [value(lazy, str(index))["level"] for index in range(3)] == [
            13, 9, 5
        ]
        lazy_computes = [
            instruction for instruction in lazy["execution"]
            if instruction["kind"] == "compute"
        ]
        assert [instruction["op"] for instruction in lazy_computes] == [
            "rescale", "mod_switch"
        ]
        assert lazy_computes[0]["attrs"] == {
            "target_level": 9,
            "target_scale_log2": 40,
        }
        assert lazy_computes[1]["attrs"] == {"target_level": 5}

        lazy_mlir = lazy_mlir_path.read_text(encoding="utf-8")
        assert "ckks.logical_init_level = 5 : i64" in lazy_mlir
        assert "ckks.levels_per_logical_level = 4 : i64" in lazy_mlir
        assert "init_level = 13 : i64" in lazy_mlir
        assert "downFactor = 4 : i64" in lazy_mlir
        for op_name, expected_duration in (("rescalec", 113),
                                           ("modswitchc", 209)):
            line = next(
                line for line in lazy_mlir.splitlines()
                if f'"ckks.{op_name}"' in line
            )
            start = int(re.search(
                r"dist.schedule_start = ([0-9]+)", line
            ).group(1))
            finish = int(re.search(
                r"dist.schedule_finish = ([0-9]+)", line
            ).group(1))
            assert finish - start == expected_duration

        for fixture, expected_error in (
            ("lazy-physical-levels-invalid-scale",
             "scale drop does not match the target physical modulus bits"),
            ("lazy-physical-levels-underflow",
             "requires a physical level below the OperatorSpec lower_bound"),
        ):
            failed_materialization = subprocess.run(
                [
                    str(args.hecate_opt),
                    str(args.source_dir / f"test/runtime-plan/{fixture}.mlir"),
                    f"-p=builtin.module(func.func({physical_levels}))",
                    "-o", str(temp / f"{fixture}.mlir"),
                ],
                cwd=args.source_dir,
                text=True,
                capture_output=True,
            )
            assert failed_materialization.returncode != 0
            assert expected_error in failed_materialization.stderr

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

        placement_1x8, placement_1x8_mlir = run_placement_pipeline(
            args.hecate_opt, args.source_dir, temp, "8", 18
        )
        verify_placement(placement_1x8, placement_1x8_mlir, [8])
        verify_rotate_costs(placement_1x8_mlir)
        placement_1x8_repeat, placement_1x8_repeat_mlir = run_placement_pipeline(
            args.hecate_opt, args.source_dir, temp, "8", 18
        )
        assert placement_1x8_repeat == placement_1x8
        assert placement_1x8_repeat_mlir == placement_1x8_mlir

        placement_2x8, placement_2x8_mlir = run_placement_pipeline(
            args.hecate_opt, args.source_dir, temp, "8x8", 28
        )
        verify_placement(placement_2x8, placement_2x8_mlir, [8, 8])

        placement_2cpu, placement_2cpu_mlir = run_placement_pipeline(
            args.hecate_opt, args.source_dir, temp, "0x0", 38
        )
        verify_placement(placement_2cpu, placement_2cpu_mlir, [0, 0])

        placement_boot, placement_boot_mlir = run_placement_pipeline(
            args.hecate_opt, args.source_dir, temp, "1", 48,
            fixture="placement-boot", function="placement_boot",
            boot_profile="test-boot",
        )
        assert placement_boot["target"]["device_counts"] == [1]
        boot_computes = [instruction for instruction in placement_boot["execution"]
                         if instruction["kind"] == "compute"]
        assert [instruction["op"] for instruction in boot_computes] == [
            "negate", "boot", "negate"
        ]
        assert [place_key(instruction["place"])
                for instruction in boot_computes] == [(0, 0), (0, -1), (0, 0)]
        boot_transfers = [
            instruction
            for phase in (placement_boot["initialization"],
                          placement_boot["execution"])
            for instruction in phase
            if instruction["kind"] == "transfer"
        ]
        assert len(boot_transfers) == 3
        assert [(place_key(instruction["sources"][0]),
                 place_key(instruction["destinations"][0]))
                for instruction in boot_transfers] == [
                    ((0, -1), (0, 0)),
                    ((0, 0), (0, -1)),
                    ((0, -1), (0, 0)),
                ]
        assert re.search(r'"ckks\.bootstrapc".*dist\.device = -1',
                         placement_boot_mlir)

        communication_profile = (
            args.source_dir /
            "test/runtime-plan/placement-communication-profile.json"
        )
        communication, communication_mlir = run_placement_pipeline(
            args.hecate_opt, args.source_dir, temp, "2", 58,
            fixture="placement-communication-model",
            function="placement_communication_model",
            communication_profile=communication_profile,
        )
        assert communication["target"]["device_counts"] == [2]
        communication_schedule = {}
        for line in communication_mlir.splitlines():
            if '"ckks.' not in line or "dist.schedule_start" not in line:
                continue
            logical_id = int(re.search(
                r"dist.logical_id = ([0-9]+)", line
            ).group(1))
            communication_schedule[logical_id] = (
                int(re.search(r"dist.schedule_start = ([0-9]+)", line).group(1)),
                int(re.search(r"dist.schedule_finish = ([0-9]+)", line).group(1)),
                int(re.search(r"dist.device = (-?[0-9]+)", line).group(1)),
            )
        # Payload: 2 components * 6 limbs * degree 16 * 8 bytes = 1536 bytes.
        # Host/device curve: 3 + ceil((1536 + 512) / 128) = 19 us.
        # Intra-rank table interpolation: 2 + ceil(1536 / 106.66...) = 17 us.
        assert communication_schedule == {
            1: (19, 1019, 0),
            2: (19, 1019, 1),
            3: (1036, 1136, 0),
        }


if __name__ == "__main__":
    main()
