# DaCapo / Hecate Compiler

DaCapo is an MLIR compiler for approximate CKKS computation. This fork emits RuntimePlan for CKKS Runtime, including physical levels, rank/device placement, explicit communication, Boot placement and memory scheduling. The old HEVM bytecode emitter, C++ interpreters and Python runner have been removed.

This README describes the current pipeline and usage. Development plans and old test records live in Git history. CKKS Runtime maintains the execution contract in its `docs/runtime-plan/` and `docs/operator-spec/` documents; update those contracts together with compiler changes.

## Build and test

The Nix environment provides a matching LLVM/MLIR/Clang 18 toolchain:

```bash
nix develop
cmake --preset nix
cmake --build --preset nix --parallel 2
ctest --test-dir build/nix --output-on-failure
```

`nix build` builds the compiler package. A manual build needs CMake 3.22 or newer, Ninja and a matching LLVM/MLIR installation. Python 3.10 or newer and `requirements.txt` are needed for model tracing. CTest also uses Python for compiler checks.

## RuntimePlan pipeline

CKKS operations have pure SSA results. `!ckks.poly` carries components, scale_log2 and the Runtime-direction level: larger levels retain more RNS moduli.

| Pass | Purpose |
| --- | --- |
| `materialize-ckks-physical-levels` | Expand logical levels to the target RNS chain before placement |
| `assign-ckks-placement` | Deterministic HEFT placement using OperatorSpec V2; decrypt/re-encrypt Boot selects Host candidates |
| `materialize-ckks-communication` | Insert explicit point-to-point `dist.transfer` copies |
| `plan-plaintext-waterline` | Lower direct Host Encode uploads to Device Prepare and check complete single-group capacity |
| `plan-runtime-memory` | Plan last-use Release and supported input reuse |
| `estimate-runtime-memory` | Report RNS object storage under sequential completion |
| `emit-runtime-plan` | Export semantic RuntimePlan V1/V2/V3/V4 as binary or JSON |

Run `hecate-opt --help` for options. Plain eager plans use V1, memory instructions use V2, execution Encode or explicit Fence uses V3, and Waterline Prepare/configuration uses V4. Format versions describe different semantics; they do not select the binary container version.

Export defaults to binary: `.runtime-plan.bin`, a packed bundle with `manifest.bin` and `data.bin`. For readable instructions use `--runtime-plan-format=json --runtime-plan-pretty=true`; export pass options are `plan-format=json pretty=true`. Binary uses `plan-format=binary`, requires `bundle-format=pack` and `pretty=false`, and has versioned little-endian `CKKSPL01` / `CKKSMF01` headers. Use separate prefixes for JSON and binary artifacts.

Bundles default to storage V2 pack. Legacy per-blob output requires both `--runtime-plan-format=json --runtime-plan-bundle-format=files` (pass: `plan-format=json bundle-format=files`). Inline Encode payloads default to a 4096-byte threshold; larger payloads are deduplicated by content ID.

For GPU plans, `--runtime-plan-device-counts=1` or `=4` enables placement and communication. `--runtime-plan-memory --runtime-plan-memory-report` adds Release/reuse and before/after per-place estimates. Eager mode keeps Encode/uploads in initialization; Release alone does not make all weights lazy. Estimates count 8-byte Host and 4-byte GPU RNS words, excluding keys, workspaces, transfer staging and allocator pools.

## Waterline plaintext preparation

`--runtime-plan-plaintext-schedule=eager|waterline` defaults to eager. Waterline requires one-rank placement and positive budgets:

- `--runtime-plan-gpu-budget-bytes`: per-card executor capacity, with reserve and dynamic objects charged together.
- `--runtime-plan-host-budget-bytes`: shared Host objects, raw buffers, CPU encoder temporaries and pinned staging.
- `--runtime-plan-pinned-budget-bytes`: shared pinned upload/communication staging.
- `--runtime-plan-raw-budget-bytes`: admitted raw read/decode data.

`--runtime-plan-reserve-bytes` covers keys, parameter tables, retained workspace, overhead and safety margin. `--runtime-plan-waterline-fraction` must be strictly between zero and one. `--runtime-plan-workspace-bytes-per-op` bounds a consumer group's foreground workspace; zero only checks objects, not total GPU memory. `--runtime-plan-prepare-backend=cpu|gpu|auto` selects complete Encode candidates; `--runtime-plan-lookahead-groups` bounds the rolling window.

Waterline runs after unchanged HEFT placement and communication, before memory planning, which is always enabled in Waterline mode. It emits one shared Device Prepare for a direct Host Encode/upload, without static batch cloning, re-encoding or automatic Fence. Do not run Encode CSE afterward. Plaintext final outputs, multiple ranks and infeasible complete groups are rejected; ordinary Encode/Transfer and explicit Fence retain their meaning.

The `.waterline.json` report checks a sequential complete-group baseline with retained objects and the configured workspace upper bound. Runtime repeats the check using backend resources, then admits complete direct-dependency groups and releases credits at actual completion. Host/pinned/raw budgets are shared across GPUs. Keys, parameters, retained caches and allocator overhead must fit the reserve; object reports do not prove total driver memory. See the parent Runtime's [`docs/runtime-plan/v4.md`](../../docs/runtime-plan/v4.md). CTest's `check-waterline.py` checks Prepare V4, sharing, unchanged foreground placement/order, no automatic Fence and single-group rejection.

## Boot planning

`--boot-placement=optimized|greedy|depth-dp` selects the Earth planner; optimized remains the default. The model artifact generator in the parent CKKS Runtime repository accepts the same option.

All three planners preserve the residual scale: after multiplication, rescale while `scale >= waterline + rescalingFactor`, consuming one Q each time. Addition raises the smaller scale with encoded one and aligns levels with ModSwitch. `--bootstrap-scale` specifies the Boot output independently of its input; zero retains the legacy `rescalingFactor` default. Boot resets consumed Q to zero and only consumes additional Q if its output itself crosses the Rescale threshold. Returns retain their remaining Q.

Optimized retains DaCapo's candidate points, bypass analysis, coverage and latency planning. Analysis and emission share these scale transitions. Its dynamic program keeps distinct scale/Q states for live values across a boundary, using the compiler profile's level-dependent latency tables. Plans record boundary/target pairs, so a value may be refreshed repeatedly and its earlier consumers keep their original input. Inputs which have already consumed Q may be refreshed at boundary zero. It is more expensive to compile than the fast planners; the candidate/bypass search is not a claim of global optimality over all DAG placements.

Greedy refreshes exhausted multiply inputs and shares refreshes with later uses. Depth-dp uses multiplicative-depth boundaries and a conservative count of unique ciphertexts crossing them, with analysis cost `O(V + E + D*L)` plus `O(V log V)` operation ordering. Its optimum applies only to this restricted boundary cost model, not arbitrary DAG placement or level-dependent latency.

Depth-dp adapts the method in [Fhelipe, PLDI 2024](https://doi.org/10.1145/3656382). The upstream reference is revision `891b3086bf6a144deebac79290801253b9cc510c`, particularly [the DP pass](https://github.com/fhelipe-compiler/fhelipe/blob/891b3086bf6a144deebac79290801253b9cc510c/backend/src/dp_bootstrapping_pass.cc) and [depth analysis](https://github.com/fhelipe-compiler/fhelipe/blob/891b3086bf6a144deebac79290801253b9cc510c/backend/src/dag_depth_info.cc). This fork does not implement its connected-component chokepoint or exhaustive shortcut subset selection.

The planners support straight-line Earth functions with ciphertext inputs and constant, multiply, add, rotate and negate operations. Scales and `rescalingFactor` must be positive, Boot output scale must be at least `waterline`, and `bootstrapLevelUpperBound - bootstrapLevelLowerBound` must leave enough Q for the chosen operations. Unsupported inputs and infeasible budgets produce errors. Optimized does not switch to either fast planner on failure.

## Manual compilation checkpoints

Generate a checkpoint explicitly with `--dacapo-precompile`. It runs the same Boot planning, scale management, Earth-to-CKKS lowering, CSE and canonicalization as the first part of `--dacapo`, then stops before physical-level materialization, placement, communication and RuntimePlan export:

```bash
build/nix/bin/hecate-opt model.earth.mlir --dacapo-precompile \
  --ckks-config=config.json --waterline=40 --bootstrap-scale=45 --boot-placement=optimized \
  --emit-bytecode -o model.ckks.mlirbc
```

Reuse that file explicitly with `--ckks-to-runtime-plan`. Pass the same `--ckks-config` and your normal `--runtime-plan-*` options for physical levels, target metadata, placement, communication, memory planning and export. In this Bash example, `runtime_options` is an array containing those options from your full compile command:

```bash
build/nix/bin/hecate-opt model.ckks.mlirbc --ckks-to-runtime-plan \
  --ckks-config=config.json "${runtime_options[@]}" \
  -o model.gpu4.mlir
```

This second pipeline skips all Earth/Boot planning passes. Use the original unplaced checkpoint for each GPU count, topology, communication profile or memory budget. The input checkpoint is left unchanged. `--dacapo` remains the complete pipeline and uses these same two stages internally.

There is no automatic lookup, freshness check or regeneration. Regenerate the checkpoint when the DSL, traced constants/weights, early pass implementation, compiler profile, Boot strategy, waterline or other early options change. Keep the generation command alongside the checkpoint. A checkpoint already fixes logical CKKS levels and scales; later target/context and physical-level options must be compatible with them. Changing early options on the resume command does not rerun those passes. Use a compatible compiler to read its bytecode.

Both lossless text MLIR and bytecode work. Bytecode is recommended for large constants. For text, omit `--emit-bytecode`, use `.mlir`, and do not elide elements or resource data. Debug snapshots and the final placed `.mlir` are not the checkpoint produced by `--dacapo-precompile`.

## Python tracing and constants

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./install.sh
source config.sh
hc-trace MLP
```

`config.sh` provides compiler functions such as `hopts` and `hbt`; execution uses CKKS Runtime. `hc.save(directory, directory, bytecode=True)` writes lossless `.mlirbc` that `hecate-opt` reads directly. `--mlir-elide-elementsattrs-if-larger=16` only reduces diagnostic MLIR; that elided text cannot be recompiled, but RuntimePlan and bundle still retain all payloads.

Runtime zero-pads short constants. Prefix ones/minus-ones remain masks; only full-slot units fold to Upscale/Negate with the encoding scale preserved. This needs parameter-managed functions, not frontend defaults. An Upscale after a single-use ciphertext/plaintext product is absorbed into the plaintext encoding scale; shared products keep their scale. Equal Encode operations share a value through CSE before placement and Waterline. Retrace sources when frontend constant rules change, and rebuild both `hecate-opt` and `HecateFrontend`.

## Papers

- DaCapo: Automatic Bootstrapping Management for Efficient Fully Homomorphic Encryption, USENIX Security 2024.
- ELASM: Error-Latency-Aware Scale Management for Fully Homomorphic Encryption, USENIX Security 2023.
- HECATE: Performance-Aware Scale Optimization for Homomorphic Encryption Compiler, CGO 2022.
