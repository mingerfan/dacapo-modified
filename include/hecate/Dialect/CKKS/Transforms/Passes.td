

#ifndef HECATE_DIALECT_CKKS_TRANSFORMS_PASSES
#define HECATE_DIALECT_CKKS_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def RemoveLevel : Pass<"remove-level", "::mlir::func::FuncOp"> {
  let summary = "Deprecated: remove CKKS levels for the retired HEVM path";
  let description = [{
  }];
}

def EmitHEVM : Pass<"emit-hevm", "::mlir::func::FuncOp"> {
  let summary = "Retired: HEVM emission is no longer supported";
  let description = [{
  }];
  let options = [
    Option<"prefix", "prefix", "std::string", /*default=*/[{""}],
    "Name prefix of output file">,
  ];

}

def EmitRuntimePlan : Pass<"emit-runtime-plan", "::mlir::func::FuncOp"> {
  let summary = "Emit a single-Host RuntimePlan V1 JSON file";
  let description = [{
    Serializes a linear CKKS function to RuntimePlan V1. This stage-two pass
    places all values and computation on Host rank 0 and emits no communication.
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
           "Largest Encode payload kept inline; larger float64 payloads use a bundle">
  ];
}

#endif
