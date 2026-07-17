#ifndef HECATE_DIST_OPS
#define HECATE_DIST_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def Dist_Dialect : Dialect {
  let name = "dist";
  let cppNamespace = "::hecate::dist";
}

class Dist_Op<string mnemonic, list<Trait> traits = []> :
    Op<Dist_Dialect, mnemonic, traits>;

def TransferOp : Dist_Op<"transfer", [SameOperandsAndResultType]> {
  let summary = "Copy one CKKS value between two physical places";
  let arguments = (ins AnyRankedTensor:$input,
                       I64Attr:$transfer_id,
                       I64Attr:$source_rank,
                       I64Attr:$source_device,
                       I64Attr:$destination_rank,
                       I64Attr:$destination_device,
                       BoolAttr:$initialization);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

#endif
