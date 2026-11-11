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

CKKS operations have single SSA results. `!ckks.poly` carries components, scale_log2 and the Runtime-direction level: larger levels retain more RNS moduli. Prepared calls can also produce and consume an opaque `!ckks.buffer`.

| Pass | Purpose |
| --- | --- |
| `materialize-ckks-physical-levels` | Expand logical levels to the target RNS chain before placement |
| `assign-ckks-placement` | Deterministic HEFT placement using OperatorSpec V2; decrypt/re-encrypt Boot selects Host candidates |
| `materialize-ckks-communication` | Insert explicit point-to-point `dist.transfer` copies |
| `plan-plaintext-waterline` | Lower direct Host Encode uploads to Device Prepare and check complete single-group capacity |
| `plan-runtime-memory` | Plan last-use Release and supported input reuse |
| `estimate-runtime-memory` | Report RNS object storage under sequential completion |
| `emit-runtime-plan` | Export semantic RuntimePlan V1–V5/V7–V8 as binary or JSON |

Run `hecate-opt --help` for options. Plain eager plans use V1, memory instructions use V2, execution Encode or explicit Fence uses V3, Waterline Prepare/configuration uses V4, prepared API calls use V7, and plans containing opaque buffers or device-resident external ciphertexts use V8. Format versions describe different semantics; they do not select the binary container version.

Export defaults to binary: `.runtime-plan.bin`, a packed bundle with `manifest.bin` and `data.bin`. For readable instructions use `--runtime-plan-format=json --runtime-plan-pretty=true`; export pass options are `plan-format=json pretty=true`. Binary uses `plan-format=binary`, requires `bundle-format=pack` and `pretty=false`, and has versioned little-endian `CKKSPL01` / `CKKSMF01` headers. Use separate prefixes for JSON and binary artifacts.

Bundles default to storage V2 pack. Legacy per-blob output requires both `--runtime-plan-format=json --runtime-plan-bundle-format=files` (pass: `plan-format=json bundle-format=files`). Inline Encode payloads default to a 4096-byte threshold; larger payloads are deduplicated by content ID.

For GPU plans, `--runtime-plan-device-counts=1` or `=4` enables placement and communication. `--runtime-plan-memory --runtime-plan-memory-report` adds Release/reuse and before/after per-place estimates. Eager mode keeps Encode/uploads in initialization; Release alone does not make all weights lazy. Estimates count 8-byte Host and 4-byte GPU RNS words, excluding keys, workspaces, transfer staging and allocator pools.

## Prepared API calls (RuntimePlan V7/V8)

`ckks.apicall` represents an externally prepared callback, not CKKS arithmetic to be planned. It takes zero or more ciphertext tensors or opaque buffers and returns exactly one ciphertext tensor or opaque buffer. Plaintexts are not ApiCall inputs; use ordinary CKKS operations for plaintext arithmetic. Ciphertext results may have three components, for example before relinearization. `callee` must be nonempty, `call_id` must be a unique nonnegative i64, and `latency_us` must be positive. HEFT uses that latency directly. See [`test/prepared-api/branching.mlir`](test/prepared-api/branching.mlir) for generic MLIR syntax and a five-node branching example.

The callback has already fixed its internal scale changes and Boot operations. This path does **not** plan them: keep physical levels/scales in `!ckks.poly<components * scale_log2 * level>`, disable Rescale coalescing, and do not request physical-level remapping. Physical level, scale and components are exported in RuntimePlan `values`. For a noninteger actual scale, keep the nominal rounded unsigned scale in `!ckks.poly` and set `runtime.scale_log2 = 40.125 : f64` on the ApiCall. Ciphertext function arguments use `earth.arg_scales = array<f64: ...>`. Actual scales are exported as float64; the compiler does not infer a callback's internal scale changes. If a prepared graph also contains an ordinary Rescale, supply its actual `runtime.scale_log2 : f64` or a physical modulus chain; export rejects an unresolved scale instead of rounding it. Arbitrary extra MLIR attributes remain unchanged in the placed IR; they are not added to the RuntimePlan schema.

External ciphertext arguments normally start on rank 0 Host. If they are already resident on a GPU, set the **function attribute** `runtime.external_inputs_device = 0 : i64` (and `earth.arg_scales = array<f64: ...>` for actual input scales). This assigns all ciphertext arguments to rank 0/device 0 before HEFT computes dependency arrival times; no Host upload is generated for local consumers. The attribute requires one rank and a valid device index in `device-counts`; plaintext arguments still start on Host. It describes real caller-owned GPU values, not a request to hide transfers. Omitting it keeps the existing Host-input behavior.

The zero-input stem calls an external, persistent prepared object that already owns its input/weights. Those objects are prepared outside this compilation; they are not implicit function arguments or compiler-generated plaintext initialization. `callee` names the callback registered by the runtime caller. The compiler does not implement or interpret that callback.

```bash
spec=test/runtime-plan/placement-operator-spec.json
build/manual/bin/hecate-opt test/prepared-api/branching.mlir \
  --ckks-to-runtime-plan --ckks-config=config.json \
  --runtime-plan-coalesce-rescales=false --runtime-plan-levels-per-logical-level=1 \
  --runtime-plan-device-counts=1 --runtime-plan-operator-spec-path="$spec" \
  --runtime-plan-target-id=dacapo-placement-test --runtime-plan-context-id=test-context \
  --runtime-plan-operator-spec-id=dacapo-placement-test-v1 \
  --runtime-plan-operator-spec-sha256="sha256:$(sha256sum "$spec" | cut -d' ' -f1)" \
  --runtime-plan-memory --runtime-plan-memory-reuse=false \
  --runtime-plan-format=json -o /tmp/prepared.mlir
```

This emits `/tmp/prepared.prepared_branching.runtime-plan.json`. Omit `--runtime-plan-format=json` for binary output. Without buffers or device-resident external ciphertexts, both use semantic `format_version: 7`; the binary container remains `CKKSPL01` version 1. Each call is an ordinary compute instruction with `op: "ApiCall"` and `attrs: {"callee": "prepared_stem", "call_id": "41"}`. `call_id` is a canonical decimal **string**, distinct from the plan's ValueIds. `inputs`, `output`, `place`, transfers and Release follow the normal RuntimePlan protocol. Opaque calls do not support input reuse. This is executable RuntimePlan, not a separate experiment schedule format.

### Shared opaque buffers

Use the exact syntax `!ckks.buffer<1234, "ckks.hoisted.v1">`: the first parameter is the declared byte count (uint64, including zero), and the second is a nonempty layout name understood by the callbacks. The compiler does not inspect the bytes or infer this size. Ordinary CKKS arithmetic still accepts only polynomial tensors.

A shared Hoist graph is `x -> H`, then `(x, H) -> rotated_x` for each suffix. Each suffix is a separate ApiCall with one ciphertext result. See [`test/prepared-api/hoist-buffer.mlir`](test/prepared-api/hoist-buffer.mlir), which ends with an opaque Q6-to-Q30 Boot call. Buffers are internal values only: they cannot be external inputs or final outputs. Cross-place buffer transfers are rejected; the current target is one GPU. There is no automatic buffer copy or layout conversion.

HEFT schedules these calls with their normal dependencies and declared latencies. Memory planning releases a shared buffer after its last consumer; ApiCalls never reuse input storage. Memory estimates count the declared buffer bytes exactly, separately from plaintext/ciphertext RNS storage. These declared bytes are not an estimate of any other workspace hidden inside a callback.

Any plan with a buffer **or a device-resident external ciphertext** uses semantic V8, including a simple resident-input arithmetic graph with no ApiCall or Buffer. Buffer JSON values have `kind: "buffer"`, canonical decimal-string `buffer_bytes`, and string `buffer_layout`; their level, scale and components are zero and `ntt` is false. Their context and place are the real plan context and producer place. Ciphertext/plaintext JSON values do not gain buffer fields. Binary V8 appends uint64 bytes and a uint32 layout string-table reference to **every** value record (zero/empty for polynomial values), giving 51-byte records. V7 and earlier binary widths are unchanged.

`DacapoPreparedApiRuntimePlan` and `DacapoPreparedApiBuffer` check JSON/binary agreement, placement, dependencies, fractional scales, shared-buffer lifetimes/byte counts, no callback reuse, and invalid types using only Python's standard library.

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

Optimized retains DaCapo's candidate points, bypass analysis, coverage and latency planning. Analysis and emission share these scale transitions. Its dynamic program retains one prefix with the lowest accumulated estimated latency per candidate boundary, using the compiler profile's level-dependent latency tables. The retained prefix carries the types and actual scales of all live values, including bypasses, into the next segment. Other prefixes are discarded even when their output states differ; this pruning can miss a better or feasible continuation. There is no multi-state mode or automatic fallback. Plans record boundary/target pairs, so a value may be refreshed repeatedly and its earlier consumers keep their original input. Inputs which have already consumed Q may be refreshed at boundary zero. It is more expensive to compile than the fast planners; the candidate/bypass search is not a claim of global optimality over all DAG placements.

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

`--dacapo`, `--dacapo-precompile` and `--pars` default to `--polynomial-optimizations=double-align`. Before planning, full-slot constants exactly equal to two multiply ciphertexts by self-addition. After scale management, alignment is absorbed into a single-use coefficient product, including through Rescale and ModSwitch chains, when the result remains below the immediate Rescale threshold. Shared intermediates and fixed scale-management operations keep their original state. Existing traces can be reused; Boot schedules and CKKS checkpoints must be regenerated.

For compiler comparisons, `none` retains the original arithmetic and `double` enables only doubling simplification. Ordinary coefficient products use the scale-management waterline for plaintext encoding; alignment absorption can raise that encoding scale to eliminate a separate alignment multiplication.

## Papers

- DaCapo: Automatic Bootstrapping Management for Efficient Fully Homomorphic Encryption, USENIX Security 2024.
- ELASM: Error-Latency-Aware Scale Management for Fully Homomorphic Encryption, USENIX Security 2023.
- HECATE: Performance-Aware Scale Optimization for Homomorphic Encryption Compiler, CGO 2022.
