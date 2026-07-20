# DaCapo / Hecate Compiler

DaCapo is an MLIR-based compiler for approximate CKKS computation. This fork
emits versioned RuntimePlan JSON for execution by CKKS Runtime and carries the
physical-level, placement, communication, and Host Boot placement logic needed
by the Poseidon CPU/GPU backends.

The old HEVM bytecode emitter, C++ interpreters, and Python runner have been
removed. RuntimePlan is the only executable output contract in this fork.

## Build And Test

The recommended environment pins LLVM, MLIR, and Clang 18.1.8 with Nix:

```bash
nix develop
cmake --preset nix
cmake --build --preset nix --parallel 2
ctest --test-dir build/nix --output-on-failure
```

The same compiler-only package can be built with:

```bash
nix build
```

The C++ compiler needs CMake 3.22 or newer, Ninja, and a matching LLVM/MLIR
installation. Python 3.10 or newer plus `requirements.txt` is needed only for
tracing benchmark models.

## RuntimePlan Pipeline

CKKS operations use pure SSA results and carry `components`, `scale_log2`, and
Runtime-direction `level` in `!ckks.poly` types. The main integration passes
are:

- `materialize-ckks-physical-levels`: expands logical levels for targets such
  as Poseidon's four-level lazy rescale;
- `assign-ckks-placement`: assigns deterministic rank/device placement from an
  OperatorSpec and places decrypt/re-encrypt Boot operations on Host;
- `materialize-ckks-communication`: inserts explicit `dist.transfer` values;
- `emit-runtime-plan`: writes RuntimePlan V1 JSON and optional plaintext bundle
  files.

Run `hecate-opt --help` for pass options. The RuntimePlan CTest covers Host,
single-rank/multi-device, multi-rank placement, lazy physical levels, Host Boot,
inline constants, and content-addressed bundles.

## Python Tracing

Install the tracing package and generate Earth MLIR with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./install.sh
source config.sh
hc-trace MLP
```

`config.sh` keeps compiler convenience functions such as `hopts` and `hbt`.
They compile traced Earth MLIR; execution is handled by CKKS Runtime rather than
the removed HEVM runner.

## Papers

- DaCapo: Automatic Bootstrapping Management for Efficient Fully Homomorphic
  Encryption, USENIX Security 2024.
- ELASM: Error-Latency-Aware Scale Management for Fully Homomorphic Encryption,
  USENIX Security 2023.
- HECATE: Performance-Aware Scale Optimization for Homomorphic Encryption
  Compiler, CGO 2022.
