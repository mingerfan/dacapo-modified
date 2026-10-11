#!/usr/bin/env python3
"""Check compiled IR/export against independently hand-calculated object bytes."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hecate-opt", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    fixtures = args.source_dir / "test/runtime-plan"
    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)

        def compile(fixture, passes, name):
            prefix = temp / name
            estimate = f"estimate-runtime-memory{{prefix={prefix} operator-spec={fixtures / 'placement-operator-spec.json'}}}"
            emit = f"emit-runtime-plan{{plan-format=json prefix={prefix} target-id=test operator-spec-id=test operator-spec-sha256=sha256:{'0' * 64} context-id=test-context}}"
            output = temp / (name + ".mlir")
            subprocess.run(
                [
                    str(args.hecate_opt),
                    str(fixtures / (fixture + ".mlir")),
                    f"-p=builtin.module(func.func({','.join(passes + [estimate, emit])}))",
                    "-o",
                    str(output),
                ],
                cwd=args.source_dir,
                check=True,
            )
            function = fixture.replace("-", "_")
            return (
                json.loads(Path(f"{prefix}.{function}.runtime-plan.json").read_text()),
                json.loads(Path(f"{prefix}.{function}.memory.json").read_text()),
                output,
            )

        ordinary, before, _ = compile("memory-host", [], "ordinary")
        released, release_report, _ = compile(
            "memory-host", ["plan-runtime-memory{enable-reuse=false}"], "release"
        )
        reused, after, optimized = compile(
            "memory-host", ["plan-runtime-memory"], "reuse"
        )
        assert (
            ordinary["format_version"] == 1
            and released["format_version"] == reused["format_version"] == 2
        )
        assert [r["peak_bytes"] for r in before["places"]] == [3328]
        assert [r["peak_bytes"] for r in release_report["places"]] == [1792]
        assert [r["peak_bytes"] for r in after["places"]] == [1792]
        assert [r["distinct_allocation_bytes"] for r in after["places"]] == [1792]
        assert [r["distinct_allocation_bytes"] for r in release_report["places"]] == [
            3328
        ]
        assert (
            after["places"][0]["retained_bytes"] == 1024
        )  # borrowed input + final output
        assert sum("reuse_input" in op for op in reused["execution"]) == 3
        assert [v["id"] for v in ordinary["values"]] == [
            v["id"] for v in reused["values"]
        ]
        assert (
            ordinary["initialization"][0]["output"]
            == reused["initialization"][0]["output"]
            == "2"
        )
        overwritten = {
            op["inputs"][0] for op in reused["execution"] if "reuse_input" in op
        }
        assert not overwritten.intersection(
            op["value"] for op in reused["execution"] if op["kind"] == "release"
        )
        peak = after["places"][0]["peak"]
        event = next(
            op
            for op in reused["execution"]
            if op["ordinal"] == peak["instruction_ordinal"]
        )
        assert event["op"] == "add_cp" and peak["plaintext_bytes"] == 256
        assert {v["value_id"] for v in peak["largest_objects"]} == {"0", "5", "6", "2"}
        rerun = subprocess.run(
            [
                str(args.hecate_opt),
                str(optimized),
                "-p=builtin.module(func.func(plan-runtime-memory))",
            ],
            cwd=args.source_dir,
            capture_output=True,
            text=True,
        )
        assert rerun.returncode != 0 and "already been planned" in rerun.stderr
        device, device_report, _ = compile(
            "memory-devices", ["plan-runtime-memory"], "devices"
        )
        assert [r["peak_bytes"] for r in device_report["places"]] == [768, 640, 0, 512]
        assert [r["retained_bytes"] for r in device_report["places"]] == [512, 0, 0, 256]
        assert [r["distinct_allocation_bytes"] for r in device_report["places"]] == [
            768,
            640,
            0,
            512,
        ]
        assert sum("reuse_input" in op for op in device["execution"]) == 1
        assert not any(
            "reuse_input" in op and op["op"] == "negate" for op in device["execution"]
        )
        assert "Rotate and Boot workspaces" in device_report["excluded"]
        levels, levels_report, _ = compile(
            "memory-levels", ["plan-runtime-memory"], "levels"
        )
        assert (
            levels_report["places"][0]["peak_bytes"] == 2048
        )  # level 5 input (1536) + level 1 output (512)
        assert levels_report["places"][0]["retained_bytes"] == 2048
        fanout, _, _ = compile("memory-final-fanout", ["plan-runtime-memory"], "fanout")
        assert not any("reuse_input" in op for op in fanout["execution"])
        assert not set(fanout["final_outputs"]).intersection(
            op["value"] for op in fanout["execution"] if op["kind"] == "release"
        )
        assert any(
            op["kind"] == "release" for op in fanout["execution"]
        )  # Release carries a Free effect.
        # Standalone estimation failure is explicit, with no empty report fallback.
        bad = subprocess.run(
            [
                str(args.hecate_opt),
                str(fixtures / "memory-host.mlir"),
                f"-p=builtin.module(func.func(estimate-runtime-memory{{prefix={temp / 'bad'} operator-spec={temp / 'missing.json'}}}))",
            ],
            cwd=args.source_dir,
            capture_output=True,
            text=True,
        )
        assert bad.returncode != 0 and "cannot open OperatorSpec" in bad.stderr
        for content in (
            '{"context": {}}',
            "{invalid json",
            '{"context":{"poly_degree":"bad"}}',
        ):
            invalid_spec = temp / "invalid.json"
            invalid_spec.write_text(content)
            invalid = subprocess.run(
                [
                    str(args.hecate_opt),
                    str(fixtures / "memory-host.mlir"),
                    f"-p=builtin.module(func.func(estimate-runtime-memory{{prefix={temp / 'invalid'} operator-spec={invalid_spec}}}))",
                ],
                cwd=args.source_dir,
                capture_output=True,
                text=True,
            )
            assert invalid.returncode == 1 and "OperatorSpec" in invalid.stderr
    print("Runtime memory planner and hand-calculated estimates passed")


if __name__ == "__main__":
    main()
