#!/usr/bin/env python3
"""Check compact/diagnostic export equivalence, exact digests and publication."""
import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hecate-opt", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="dacapo-plan-io-") as directory:
        temp = Path(directory)
        # Convert once and reuse the same IR for each output format.
        ir = temp / "bundle.mlir"
        subprocess.run([str(args.hecate_opt),
                        str(args.source_dir / "test/runtime-plan/bundle-reuse.mlir"),
                        "-p=builtin.module(func.func(convert-earth-to-ckks))", "-o", str(ir)],
                       cwd=args.source_dir, check=True)
        prefix = temp / "out"

        def emit(pretty):
            options = (f"prefix={prefix} target-id=test operator-spec-id=test "
                       f"operator-spec-sha256=sha256:{'0' * 64} context-id=test-context "
                       f"pretty={'true' if pretty else 'false'} report-io=true")
            result = subprocess.run([str(args.hecate_opt), str(ir),
                                     f"-p=builtin.module(func.func(emit-runtime-plan{{{options}}}))",
                                     "-o", "/dev/null"], cwd=args.source_dir,
                                    check=True, capture_output=True, text=True)
            path = Path(f"{prefix}.bundle_reuse.runtime-plan.json")
            raw = path.read_bytes()
            plan = json.loads(raw)
            manifest = Path(f"{prefix}.bundle_reuse.bundle/manifest.json").read_bytes()
            assert plan["plaintext_bundle"]["manifest_sha256"] == "sha256:" + hashlib.sha256(manifest).hexdigest()
            report = json.loads(next(line.removeprefix("runtime-plan-io ")
                                     for line in result.stderr.splitlines() if line.startswith("runtime-plan-io ")))
            assert report["plan_bytes"] == len(raw)
            assert report["plan_sha256"] == "sha256:" + hashlib.sha256(raw).hexdigest()
            assert report["unique_blobs"] == 1 and report["duplicate_hits"] == 1
            assert report["bundle_bytes"] == 600 * 8
            assert report["manifest_bytes"] == len(manifest)
            assert report["pretty"] == pretty
            assert not list(temp.rglob("*.tmp"))
            return plan, raw

        compact, compact_bytes = emit(False)
        diagnostic, diagnostic_bytes = emit(True)
        assert compact == diagnostic
        assert len(compact_bytes) < len(diagnostic_bytes)
        assert compact_bytes.count(b"\n") == 1
        assert emit(False)[1] == compact_bytes
        # A failed publication must preserve the existing entry point.
        output = Path(f"{prefix}.bundle_reuse.runtime-plan.json")
        temporary = Path(str(output) + ".tmp")
        temporary.mkdir()
        result = subprocess.run([str(args.hecate_opt), str(ir),
                                 f"-p=builtin.module(func.func(emit-runtime-plan{{prefix={prefix} target-id=test operator-spec-id=test operator-spec-sha256=sha256:{'0' * 64} context-id=test-context}}))",
                                 "-o", "/dev/null"], cwd=args.source_dir,
                                capture_output=True, text=True)
        assert result.returncode != 0 and "cannot open output file" in result.stderr
        assert output.read_bytes() == compact_bytes
        temporary.rmdir()
        bundle = Path(f"{prefix}.bundle_reuse.bundle")
        original_manifest = (bundle / "manifest.json").read_bytes()
        blob = next((bundle / "data").glob("*.bin"))
        original_blob = blob.read_bytes()
        blob.write_bytes(bytes([original_blob[0] ^ 1]) + original_blob[1:])
        result = subprocess.run([str(args.hecate_opt), str(ir),
                                 f"-p=builtin.module(func.func(emit-runtime-plan{{prefix={prefix} target-id=test operator-spec-id=test operator-spec-sha256=sha256:{'0' * 64} context-id=test-context}}))",
                                 "-o", "/dev/null"], cwd=args.source_dir,
                                capture_output=True, text=True)
        assert result.returncode != 0 and "existing bundle blob SHA-256 mismatch" in result.stderr
        assert output.read_bytes() == compact_bytes
        assert (bundle / "manifest.json").read_bytes() == original_manifest
        assert not list(temp.rglob("*.tmp"))
        blob.write_bytes(original_blob)
        # Fail after the first Encode wrote a blob: no final artifact survives.
        bad = temp / "bad.mlir"
        text = ir.read_text()
        index = text.rfind('"ckks.encode"')
        assert index >= 0
        bad.write_text(text[:index] + text[index:].replace('"ckks.encode"', '"test.unsupported"', 1))
        bad_prefix = temp / "failed"
        result = subprocess.run([str(args.hecate_opt), str(bad), "--allow-unregistered-dialect",
                                 f"-p=builtin.module(func.func(emit-runtime-plan{{prefix={bad_prefix} target-id=test operator-spec-id=test operator-spec-sha256=sha256:{'0' * 64} context-id=test-context}}))",
                                 "-o", "/dev/null"], cwd=args.source_dir,
                                capture_output=True, text=True)
        assert result.returncode != 0 and "unsupported operation" in result.stderr
        assert not list(temp.glob("failed.*"))
    print("RuntimePlan I/O checks passed")


if __name__ == "__main__":
    main()
