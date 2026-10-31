#!/usr/bin/env python3
"""Measure the reduction pass on already managed 10k/100k/1m instruction IR."""

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hecate-opt", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--widths", type=int, nargs="+", default=[2500, 25000, 250000])
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    moduli = [(1 << 30) - (i + 1) * 32768 + 1 for i in range(10)]
    actual = 80 - math.log2(moduli[-1])
    ct = "tensor<1x!ckks.poly<2 * 40 * 9>>"
    pt = "tensor<1x!ckks.poly<1 * 40 * 9>>"
    raw = "tensor<1x!ckks.poly<2 * 80 * 9>>"
    low = f"tensor<1x!ckks.poly<2 * {math.ceil(actual)} * 8>>"
    fixture = (
        args.source_dir / "test/runtime-plan/lazy-physical-level-operator-spec.json"
    )
    spec = json.loads(fixture.read_text())
    spec["context"]["rns_moduli"] = moduli
    for op in ("add_cc", "rescale"):
        spec["operators"][op].update(supported=True, latency_us_by_level=[10] * 10)
    (out / "spec.json").write_text(json.dumps(spec))
    rows = []
    for width in args.widths:
        src = out / f"{width}.input.mlir"
        dest = out / f"{width}.mlir"
        with src.open("w") as f:
            f.write(
                f"module {{ func.func @reduce(%x: {ct}) -> {low} attributes {{"
                "ckks.rns_moduli = array<i64: " + ", ".join(map(str, moduli)) + ">} {\n"
            )
            previous = "%r0"
            for i in range(width):
                f.write(
                    f'%p{i} = "ckks.encode"() {{payload = dense<0.125> : tensor<8xf64>, runtime.scale_log2 = 4.000000e+01 : f64}} : () -> {pt}\n'
                )
                f.write(
                    f'%m{i} = "ckks.mulcp"(%x, %p{i}) {{runtime.scale_log2 = 8.000000e+01 : f64}} : ({ct}, {pt}) -> {raw}\n'
                )
                f.write(
                    f'%r{i} = "ckks.rescalec"(%m{i}) {{runtime.scale_log2 = {actual:.17e} : f64}} : ({raw}) -> {low}\n'
                )
                if i:
                    f.write(
                        f'%a{i} = "ckks.addcc"({previous}, %r{i}) {{runtime.scale_log2 = {actual:.17e} : f64}} : ({low}, {low}) -> {low}\n'
                    )
                    previous = f"%a{i}"
            f.write(f"return {previous} : {low}\n}}}}\n")
        pipeline = (
            "builtin.module(func.func(coalesce-reduction-rescales{"
            + f"operator-spec={out}/spec.json"
            + "}))"
        )
        command = [
            "/usr/bin/time",
            "-v",
            "-o",
            str(out / f"{width}.time.log"),
            str(args.hecate_opt.resolve()),
            str(src),
            f"--ckks-config={args.source_dir}/config.json",
            f"--pass-pipeline={pipeline}",
            "--mlir-timing",
            "--mlir-disable-threading",
            "-o",
            str(dest),
        ]
        start = time.monotonic()
        with (out / f"{width}.pass.log").open("w") as log:
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, timeout=600
            )
        assert result.returncode == 0, (out / f"{width}.pass.log").read_text()[-6000:]
        elapsed = time.monotonic() - start
        count = 0
        with dest.open() as f:
            for line in f:
                count += '"ckks.rescalec"' in line
        assert count == 1, count
        resource = (out / f"{width}.time.log").read_text()
        rss = int(
            re.search(r"Maximum resident set size \(kbytes\): (\d+)", resource)[1]
        )
        timing = (out / f"{width}.pass.log").read_text()
        match = re.search(
            r"(?m)^\s*([\d.]+)\s+\([^)]*\)\s+CoalesceReductionRescales\s*$", timing
        )
        assert match, timing
        rows.append(
            dict(
                input_instructions=4 * width - 1,
                products=width,
                output_rs=count,
                wall_seconds=elapsed,
                pass_seconds=float(match[1]),
                max_rss_kib=rss,
            )
        )
        (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    main()
