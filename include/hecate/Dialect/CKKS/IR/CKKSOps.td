#ifndef HECATE_CKKS_OPS
#define HECATE_CKKS_OPS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
// Provides SameOperandsAndResultType/Shape traits only.
include "mlir/Interfaces/InferTypeOpInterface.td"
include "hecate/Dialect/CKKS/IR/PolyTypeInterface.td"

def CKKS_Dialect : Dialect {
  let name = "ckks";
  let cppNamespace = "::hecate::ckks";
  let useDefaultTypePrinterParser = 1;
}

def PolyType : TypeDef<CKKS_Dialect, "Poly", [
    DeclareTypeInterfaceMethods<PolyTypeInterface>,
    DeclareTypeInterfaceMethods<MemRefElementTypeInterface>]> {
  let summary = "CKKS polynomial type with components, scale and level";
  let description = [{
    Represents a CKKS plaintext or ciphertext. Plaintexts have one component;
    ciphertexts have at least two. scale_log2 is the binary scale exponent.
  }];
  let mnemonic = "poly";
  let parameters = (ins "unsigned":$components, "unsigned":$scale_log2,
                         "unsigned":$level);
  let assemblyFormat = "`<` $components `*` $scale_log2 `*` $level `>`";
  let genVerifyDecl = 0;
}

def PolyTensor : TensorOf<[PolyType]>;

class CKKS_Op<string mnemonic, list<Trait> traits = []> :
    Op<CKKS_Dialect, mnemonic, traits> {
  let results = (outs PolyTensor:$result);
}

// Constants are kept as real MLIR attributes. The legacy integer .cst index is
// intentionally not part of the RuntimePlan path.
def EncodeOp : CKKS_Op<"encode"> {
  let arguments = (ins AnyAttr:$payload);
  let hasVerifier = 1;
}

def RotateCOp : CKKS_Op<"rotatec", [SameOperandsAndResultType]> {
  let arguments = (ins PolyTensor:$src, DenseI64ArrayAttr:$offset);
  let hasVerifier = 1;
}

def NegateCOp : CKKS_Op<"negatec", [SameOperandsAndResultType]> {
  let arguments = (ins PolyTensor:$src);
  let hasVerifier = 1;
}

def RelinearizeOp : CKKS_Op<"relinearize", [SameOperandsAndResultShape]> {
  let arguments = (ins PolyTensor:$src);
  let hasVerifier = 1;
}

def RescaleCOp : CKKS_Op<"rescalec", [SameOperandsAndResultShape]> {
  let arguments = (ins PolyTensor:$src);
  let hasVerifier = 1;
}

def ModswitchCOp : CKKS_Op<"modswitchc", [SameOperandsAndResultShape]> {
  let arguments = (ins PolyTensor:$src, I64Attr:$downFactor);
  let hasVerifier = 1;
}

// UpscaleC is an intermediate operation and must be eliminated by
// convert-upscale-to-mulcp before RuntimePlan export.
def UpscaleCOp : CKKS_Op<"upscalec", [SameOperandsAndResultShape]> {
  let arguments = (ins PolyTensor:$src, I64Attr:$upFactor);
  let hasVerifier = 1;
}

def BootstrapCOp : CKKS_Op<"bootstrapc", [SameOperandsAndResultShape]> {
  let arguments = (ins PolyTensor:$src);
  let hasVerifier = 1;
}

def AddCCOp : CKKS_Op<"addcc", [Commutative, SameOperandsAndResultType]> {
  let arguments = (ins PolyTensor:$lhs, PolyTensor:$rhs);
  let hasVerifier = 1;
}

def AddCPOp : CKKS_Op<"addcp", [SameOperandsAndResultShape]> {
  let arguments = (ins PolyTensor:$lhs, PolyTensor:$rhs);
  let hasVerifier = 1;
}

def MulCCOp : CKKS_Op<"mulcc", [Commutative, SameOperandsAndResultShape]> {
  let arguments = (ins PolyTensor:$lhs, PolyTensor:$rhs);
  let hasVerifier = 1;
}

def MulCPOp : CKKS_Op<"mulcp", [SameOperandsAndResultShape]> {
  let arguments = (ins PolyTensor:$lhs, PolyTensor:$rhs);
  let hasVerifier = 1;
}

#endif
