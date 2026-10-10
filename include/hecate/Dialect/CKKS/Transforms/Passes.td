

#ifndef HECATE_DIALECT_CKKS_TRANSFORMS_PASSES
#define HECATE_DIALECT_CKKS_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def EmitRuntimePlan : Pass<"emit-runtime-plan", "::mlir::func::FuncOp"> {
  let summary = "Emit a Host or placed RuntimePlan V1 JSON file";
  let description = [{
    Serializes a linear CKKS function to RuntimePlan V1. Unplaced functions use
    Host rank 0. Placed functions preserve rank/device assignments and explicit
    dist.transfer operations.
  }];
  let options = [
    Option<"prefix", "prefix", "std::string", /*default=*/[{""}],
           "Output filename prefix">,
    Option<"planId", "plan-id", "std::string", /*default=*/[{"0"}],
           "RuntimePlan decimal plan id">,
    Option<"targetId", "target-id", "std::string", /*default=*/[{""}],
           "Runtime target id">,
    Option<"capabilityVersion", "capability-version", "int64_t",
           /*default=*/[{1}], "Runtime target capability version">,
    Option<"operatorSpecId", "operator-spec-id", "std::string",
           /*default=*/[{""}], "OperatorSpec id">,
    Option<"operatorSpecVersion", "operator-spec-version", "int64_t",
           /*default=*/[{1}], "OperatorSpec version">,
    Option<"operatorSpecSha256", "operator-spec-sha256", "std::string",
           /*default=*/[{""}], "SHA-256 of the exact OperatorSpec bytes">,
    Option<"contextId", "context-id", "std::string", /*default=*/[{""}],
           "CKKS context id">,
    Option<"deviceCount", "device-count", "int64_t", /*default=*/[{0}],
           "Device count for the single rank target">,
    Option<"ntt", "ntt", "bool", /*default=*/[{true}],
           "NTT state written to every ValueDesc">,
    Option<"bootProfile", "boot-profile", "std::string",
           /*default=*/[{""}], "OperatorSpec boot profile id">,
    Option<"bootImplementation", "boot-implementation", "std::string",
           /*default=*/[{"native"}],
           "Boot implementation: native or decrypt_reencrypt">,
    Option<"inlinePayloadMaxBytes", "inline-payload-max-bytes", "int64_t",
           /*default=*/[{4096}],
           "Largest Encode payload kept inline; larger float64 payloads use a bundle">,
    Option<"pretty", "pretty", "bool", /*default=*/[{false}],
           "Indent RuntimePlan JSON for diagnostics (default: compact)">,
    Option<"reportIo", "report-io", "bool", /*default=*/[{false}],
           "Report RuntimePlan export timings and bytes to stderr">
  ];
}

def MaterializePhysicalLevels : Pass<"materialize-ckks-physical-levels",
                                     "::mlir::func::FuncOp"> {
  let summary = "Map logical CKKS levels to a target's physical RNS levels";
  let description = [{
    Keeps Earth scale analysis in logical levels, then expands each consumed
    logical level into a fixed number of physical RNS levels before placement.
    The target OperatorSpec supplies the physical chain bounds and modulus bit
    widths used to validate Rescale operations.
  }];
  let options = [
    Option<"operatorSpecPath", "operator-spec", "std::string",
           /*default=*/[{""}], "Target OperatorSpec V1 or V2 JSON">,
    Option<"levelsPerLogicalLevel", "levels-per-logical-level", "int64_t",
           /*default=*/[{4}],
           "Physical RNS levels consumed by one logical level">
  ];
}

def AssignPlacement : Pass<"assign-ckks-placement", "::mlir::func::FuncOp"> {
  let summary = "Assign CKKS compute operations to a static rank/place topology";
  let description = [{
    Uses deterministic HEFT list scheduling. Operator latency comes from an
    OperatorSpec V2 file. Communication can use a payload-aware profile with
    rate limits and optional payload/rate interpolation points, or legacy fixed
    intra-rank and inter-rank costs.
  }];
  let options = [
    Option<"deviceCounts", "device-counts", "std::string", /*default=*/[{""}],
           "Per-rank device counts; all-zero values select CPU Host ranks">,
    Option<"operatorSpecPath", "operator-spec", "std::string", /*default=*/[{""}],
           "OperatorSpec V2 JSON used for compute latency">,
    Option<"bootProfile", "boot-profile", "std::string", /*default=*/[{""}],
           "OperatorSpec V2 Boot profile used for Boot latency">,
    Option<"communicationProfilePath", "communication-profile", "std::string",
           /*default=*/[{""}],
           "Payload-aware communication profile JSON">,
    Option<"intraRankCommunicationCost", "intra-rank-communication-cost",
           "int64_t", /*default=*/[{1000}],
           "Fixed point-to-point cost between different places in one rank">,
    Option<"interRankCommunicationCost", "inter-rank-communication-cost",
           "int64_t", /*default=*/[{10000}],
           "Fixed point-to-point cost between ranks">
  ];
}

def MaterializeCommunication : Pass<"materialize-ckks-communication",
                                    "::mlir::func::FuncOp"> {
  let summary = "Materialize cross-place CKKS operands as point-to-point transfers";
  let description = [{
    Rewrites every cross-place SSA edge through a dist.transfer operation and
    reuses one physical copy for all consumers at the same destination.
  }];
}

def PlanPlaintextStreaming : Pass<"plan-plaintext-streaming", "::mlir::func::FuncOp"> {
  let dependentDialects = ["hecate::dist::DistDialect"];
  let summary = "Schedule bounded local plaintext batches with Fence";
  let options = [
    Option<"operatorSpecPath", "operator-spec", "std::string", [{""}], "Physical OperatorSpec JSON">,
    Option<"prefix", "prefix", "std::string", [{""}], "Report filename prefix">,
    Option<"gpuBudget", "gpu-budget-bytes", "uint64_t", [{0}], "Per-GPU RNS object budget after reserves">,
    Option<"hostBudget", "host-budget-bytes", "uint64_t", [{0}], "Host RNS object budget">,
    Option<"pinnedBudget", "pinned-budget-bytes", "uint64_t", [{0}], "Batch upload staging budget">,
    Option<"rawBudget", "raw-budget-bytes", "uint64_t", [{0}], "One raw read plus decoded slots budget">,
    Option<"workspaceBytes", "workspace-bytes-per-op", "uint64_t", [{0}], "Additional cumulative per-GPU compute reservation">,
    Option<"prefetchBytes", "prefetch-bytes", "uint64_t", [{0}], "Move batch preparations ahead of compute, bounded by encoded bytes">
  ];
}

def PlanRuntimeMemory : Pass<"plan-runtime-memory", "::mlir::func::FuncOp"> {
  let dependentDialects = ["hecate::dist::DistDialect"];
  let summary = "Mark safe input reuse and release values after their last runtime use";
  let options = [
    Option<"enableReuse", "enable-reuse", "bool", [{true}],
           "Enable supported in-place operations; false plans only Release">
  ];
}

def EstimateRuntimeMemory : Pass<"estimate-runtime-memory", "::mlir::func::FuncOp"> {
  let summary = "Report per-place object peaks under sequential completion";
  let options = [
    Option<"prefix", "prefix", "std::string", [{""}], "Output filename prefix">,
    Option<"operatorSpecPath", "operator-spec", "std::string", [{""}], "OperatorSpec JSON path">,
    Option<"reportTag", "report-tag", "std::string", [{""}], "Optional report filename suffix">
  ];
}

#endif
