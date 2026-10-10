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

For large Earth graphs, add `--boot-placement=depth-dp` to a `--dacapo` command
with the existing CKKS profile and RuntimePlan options. The default remains
`--boot-placement=optimized`. The integration script
`integrations/dacapo/generate_model_artifacts.py` accepts the same option.

`depth-dp` adapts Fhelipe's multiplicative-depth boundary dynamic programming
([PLDI 2024, sections 6.2-6.3](https://doi.org/10.1145/3656382)). The upstream
source consulted is revision `891b3086bf6a144deebac79290801253b9cc510c`, specifically
[`DpBootstrappingCostFunction` and `GetBootstrappingFrontiers`](https://github.com/fhelipe-compiler/fhelipe/blob/891b3086bf6a144deebac79290801253b9cc510c/backend/src/dp_bootstrapping_pass.cc)
and [`DagDepthInfo`](https://github.com/fhelipe-compiler/fhelipe/blob/891b3086bf6a144deebac79290801253b9cc510c/backend/src/dag_depth_info.cc).
This is an independent adaptation for Earth/CKKS, rather than an import of
Fhelipe's tensor compiler. It minimizes a conservative count of unique
ciphertexts crossing boundaries, delays unary rotate/negate operations to their
earliest consumer while keeping addition reductions early to narrow frontiers,
and omits shortcut refreshes only when their remaining budget covers the entire
next segment. It does not implement
upstream's connected-component chokepoint selection, exhaustive shortcut subset
selection, or a level-dependent latency model. It is optimal only for its
restricted boundary cost model, not for arbitrary DAG placement.

The planner takes `O(V + E + D*L)` analysis time, plus `O(V log V)` to order
operations by depth; `D` is maximum multiplicative depth and `L` is the usable
level budget. It does not clone functions, run bypass simulations, or use an
external solver. Boot outputs consume one level when restored to the waterline;
the DP accounts for this rather than treating them as fresh inputs.

`--boot-placement=greedy` provides a lazy baseline: refresh only exhausted
multiply inputs and share that refresh with later uses. Both fast strategies
rescale each multiplication to the requested waterline. They can add more Boot
and rescale operations than DaCapo's latency-aware search.
They accept straight-line Earth functions with ciphertext inputs and constant,
multiply, add, rotate, and negate operations. The profile must satisfy
`0 < waterline <= rescalingFactor` and
`bootstrapLevelUpperBound - bootstrapLevelLowerBound >= 2`; unsupported inputs
produce a compiler error.

Runtime zero-pads short plaintext payloads. Prefix constants of ones or minus
ones therefore remain masks; only full-slot units can fold to Upscale/Negate,
which must retain the plaintext's encoding scale. This fold requires a
parameter-managed function: frontend defaults cannot determine the eventual
runtime slot capacity. Rebuild both `hecate-opt` and `HecateFrontend` when
updating this rule, and retrace sources exported with the old frontend.

An Upscale after a single-use ciphertext/plaintext product is absorbed into
the plaintext's encoding scale. Its payload, including zero-padded masks,
and the result's scale and level are preserved. Shared products keep their
original scale. This avoids generating a second multiplication by a full-slot
unit for scale normalization. Remaining equal Encode operations share a value
through CSE before placement, communication, and memory planning; encodings
with different payloads, shapes, scales, or levels stay distinct.

For diagnostic MLIR output, `--mlir-elide-elementsattrs-if-larger=16` reduces
disk use. That printed MLIR cannot be recompiled because constants are elided;
RuntimePlan JSON and its plaintext bundle still contain the complete payloads.

Plaintext bundles default to storage V2: one `data.bin` with byte offsets in
`manifest.json`. Use `--runtime-plan-bundle-format=files` (or the export pass
option `bundle-format=files`) for the legacy per-blob layout. This storage version
is independent of RuntimePlan V1/V2/V3; older readers require the legacy layout.

For large native traces, `hc.save(directory, directory, bytecode=True)` writes
lossless `.mlirbc` bytecode, which `hecate-opt` reads directly. Constants are
retained; no external serialization dependency is needed. Frontend source
locations use direct Python frame lookup to avoid a full stack scan per operation.

For GPU plans, `--runtime-plan-device-counts=1` or `=4` enables placement and
communication. Add `--runtime-plan-memory --runtime-plan-memory-report` for
Release, supported input reuse, and per-place estimates before and after memory
planning. Estimates count RNS object storage (8-byte host / 4-byte GPU words);
keys, operator workspaces, transfer staging and allocator pools are excluded.
Encode and its transfers still run in initialization, so Release does not make
all weight plaintexts lazy. The placement ready queue, iterative priorities,
merged occupied ranges, last-use release buckets and deferred peak reconstruction
avoid repeated full-graph scans in these passes.

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
