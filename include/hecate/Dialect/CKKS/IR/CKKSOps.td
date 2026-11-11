#ifndef HECATE_CKKS_OPS
#define HECATE_CKKS_OPS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
// Provides SameOperandsAndResultType/Shape traits only.
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
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

def BufferType : TypeDef<CKKS_Dialect, "Buffer"> {
  let summary = "Opaque buffer with a byte count and layout identifier";
  let description = [{
    Represents non-polynomial data owned by a prepared task. The byte count is
    an unsigned 64-bit value, including zero. The layout identifier is nonempty.
    Syntax: !ckks.buffer<1234, "ckks.hoisted.v1">.
  }];
  let mnemonic = "buffer";
  let parameters = (ins "uint64_t":$bytes, StringRefParameter<>:$layout);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def PolyTensor : TensorOf<[PolyType]>;
def ApiCallValue : AnyTypeOf<[PolyTensor, BufferType]>;

class CKKS_Op<string mnemonic, list<Trait> traits = []> :
    Op<CKKS_Dialect, mnemonic, traits> {
  let results = (outs PolyTensor:$result);
}

// A prepared callback is an opaque task: its internal scale/Boot decisions are
// already fixed. Extra attributes carry physical metadata without scale inference.
def ApiCallOp : CKKS_Op<"apicall"> {
  let summary = "Invoke one externally prepared CKKS task";
  let description = [{
    Polynomial inputs/results must be ciphertexts, not plaintexts.
    The optional runtime.scale_log2 attribute records the actual output scale
    exponent as a finite f64 in [0, 1023]. It does not constrain input scales.
  }];
  let arguments = (ins Variadic<ApiCallValue>:$inputs, StrAttr:$callee,
                       I64Attr:$call_id, I64Attr:$latency_us);
  let results = (outs ApiCallValue:$result);
  let hasVerifier = 1;
}

// Constants are kept as real MLIR attributes. The legacy integer .cst index is
// intentionally not part of the RuntimePlan path.
// Payload and result type fully determine the encoded plaintext before
// placement. Sharing it is safe; transfers and Release are introduced later.
def EncodeOp : CKKS_Op<"encode", [Pure]> {
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

def AddCCOp : CKKS_Op<"addcc", [Commutative, SameOperandsAndResultShape]> {
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
