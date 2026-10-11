#!/usr/bin/env python3
"""Native frontend text/bytecode exports must compile to identical plans."""
import argparse
import ctypes as C
import json
from pathlib import Path
import subprocess
import tempfile


def export(library, destination):
    lib = C.CDLL(str(library))
    lib.init.restype = C.c_void_p
    lib.createFunc.argtypes = [C.c_void_p, C.c_char_p, C.POINTER(C.c_int), C.c_size_t, C.c_char_p, C.c_size_t]
    lib.createFunc.restype = C.c_size_t
    lib.initFunc.argtypes = [C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t), C.c_size_t]
    lib.createConstant.argtypes = [C.c_void_p, C.POINTER(C.c_double), C.c_int64, C.c_char_p, C.c_size_t]
    lib.createConstant.restype = C.c_size_t
    lib.createBinary.argtypes = [C.c_void_p, C.c_size_t, C.c_size_t, C.c_size_t, C.c_char_p, C.c_size_t]
    lib.createBinary.restype = C.c_size_t
    lib.createRotation.argtypes = [C.c_void_p, C.c_size_t, C.c_int64, C.c_char_p, C.c_size_t]
    lib.createRotation.restype = C.c_size_t
    lib.setOutput.argtypes = [C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t), C.c_size_t]
    lib.save.argtypes = [C.c_void_p, C.c_char_p, C.c_char_p]
    lib.save.restype = C.c_char_p
    lib.finalize.argtypes = [C.c_void_p]
    context = lib.init()
    location = b"frontend-bytecode-probe.py"
    function = lib.createFunc(context, b"bytecode_probe", (C.c_int * 1)(0), 1, location, 1)
    arguments = (C.c_size_t * 1)()
    lib.initFunc(context, function, arguments, 1)
    mask = lib.createConstant(context, (C.c_double * 3)(1, 1, 1), 3, location, 2)
    weight = lib.createConstant(context, (C.c_double * 5)(0.1, -2.3, 1.23456789012345, 0, 3.7), 5, location, 3)
    masked = lib.createBinary(context, 8, arguments[0], mask, location, 4)
    weighted = lib.createBinary(context, 8, masked, weight, location, 5)
    rotated = lib.createRotation(context, weighted, 1, location, 6)
    result = lib.createBinary(context, 8, weighted, rotated, location, 7)
    lib.setOutput(context, function, (C.c_size_t * 1)(result), 1)
    saved = lib.save(context, str(destination.parent).encode(), str(destination).encode())
    assert saved == str(destination).encode()
    lib.finalize(context)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", type=Path, required=True)
    parser.add_argument("--hecate-opt", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="dacapo-bytecode-") as directory:
        root = Path(directory)
        config = json.loads((args.source_dir / "config.json").read_text())
        config.update(polynomialDegree=16, rescalingFactor=120,
                      bootstrapLevelLowerBound=1, bootstrapLevelUpperBound=4,
                      levelLowerBound=1, levelUpperBound=4)
        profile = root / "config.json"
        profile.write_text(json.dumps(config))
        plans = []
        for extension in (".mlir", ".mlirbc"):
            target = root / extension[1:]
            target.mkdir()
            source = target / ("source" + extension)
            export(args.frontend, source)
            if extension == ".mlirbc":
                assert source.read_bytes()[:4] == b"ML\xefR"
            output = target / "result.mlir"
            subprocess.run([
                str(args.hecate_opt), "--dacapo", "--boot-placement=depth-dp",
                f"--ckks-config={profile}", "--waterline=40", "--mlir-disable-threading",
                "--runtime-plan-id=1", "--runtime-plan-format=json", "--runtime-plan-target-id=test",
                "--runtime-plan-context-id=test", "--runtime-plan-operator-spec-id=test",
                "--runtime-plan-operator-spec-version=1",
                "--runtime-plan-operator-spec-sha256=sha256:" + "0" * 64,
                "--runtime-plan-boot-profile=test", "--runtime-plan-boot-implementation=native",
                str(source), "-o", str(output),
            ], capture_output=True, text=True, check=True)
            plans.append(json.loads((target / "result._hecate_bytecode_probe.runtime-plan.json").read_text()))
        assert plans[0] == plans[1]
        assert any(step["payload"].get("values") == [1.0, 1.0, 1.0]
                   for step in plans[0]["initialization"])
    print("Native frontend text/bytecode plans and prefix masks match")


if __name__ == "__main__":
    main()
