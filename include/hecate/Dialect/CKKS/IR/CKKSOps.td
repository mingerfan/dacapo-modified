#ifndef HECATE_CKKS_OPS
#define HECATE_CKKS_OPS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
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
    Op<CKKS_Dialect, mnemonic, !listconcat(traits, [
      DeclareOpInterfaceMethods<DestinationStyleOpInterface>,
      DeclareOpInterfaceMethods<InferTypeOpInterface>])> {
  let results = (outs PolyTensor);
  string arithClassDefinition = [{
    ::mlir::MutableOperandRange $cppClass::getDpsInitsMutable() {
      return {getOperation(), 0, 1};
    }
  }];
  let extraClassDefinition = arithClassDefinition;
}

// Constants are kept as real MLIR attributes. The legacy integer .cst index is
// intentionally not part of the RuntimePlan path.
def EncodeOp : CKKS_Op<"encode"> {
  let arguments = (ins PolyTensor:$dst, AnyAttr:$payload);
  let results = (outs PolyTensor);

  code extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(::mlir::TypeRange lhs,
                                        ::mlir::TypeRange rhs);
  }];

  code extraClassDefinition = arithClassDefinition # [{
    bool $cppClass::isCompatibleReturnTypes(::mlir::TypeRange lhs,
                                            ::mlir::TypeRange rhs) {
      return rhs.back().dyn_cast<PolyTypeInterface>().getComponents() == 1;
    }
  }];
}

def RotateCOp : CKKS_Op<"rotatec", [SameOperandsAndResultType]> {
  let arguments = (ins PolyTensor:$dst, PolyTensor:$src,
                       DenseI64ArrayAttr:$offset);
}

def NegateCOp : CKKS_Op<"negatec", [SameOperandsAndResultType]> {
  let arguments = (ins PolyTensor:$dst, PolyTensor:$src);
}

def RelinearizeOp : CKKS_Op<"relinearize", [SameOperandsAndResultShape]> {
  let arguments = (ins PolyTensor:$dst, PolyTensor:$src);
}

def RescaleCOp : CKKS_Op<"rescalec", [SameOperandsAndResultShape]> {
  let arguments = (ins PolyTensor:$dst, PolyTensor:$src);
}

def ModswitchCOp : CKKS_Op<"modswitchc", [SameOperandsAndResultShape]> {
  let arguments = (ins PolyTensor:$dst, PolyTensor:$src,
                       I64Attr:$downFactor);
}

// UpscaleC is an intermediate operation and must be eliminated by
// convert-upscale-to-mulcp before RuntimePlan export.
def UpscaleCOp : CKKS_Op<"upscalec", [SameOperandsAndResultShape]> {
  let arguments = (ins PolyTensor:$dst, PolyTensor:$src, I64Attr:$upFactor);
}

def BootstrapCOp : CKKS_Op<"bootstrapc", [SameOperandsAndResultShape]> {
  let arguments = (ins PolyTensor:$dst, PolyTensor:$src);
}

def AddCCOp : CKKS_Op<"addcc", [Commutative, SameOperandsAndResultType]> {
  let arguments = (ins PolyTensor:$dst, PolyTensor:$lhs, PolyTensor:$rhs);
}

def AddCPOp : CKKS_Op<"addcp", [Commutative, SameOperandsAndResultShape]> {
  let arguments = (ins PolyTensor:$dst, PolyTensor:$lhs, PolyTensor:$rhs);
}

def MulCCOp : CKKS_Op<"mulcc", [Commutative, SameOperandsAndResultShape]> {
  let arguments = (ins PolyTensor:$dst, PolyTensor:$lhs, PolyTensor:$rhs);
}

def MulCPOp : CKKS_Op<"mulcp", [Commutative, SameOperandsAndResultShape]> {
  let arguments = (ins PolyTensor:$dst, PolyTensor:$lhs, PolyTensor:$rhs);
}

#endif
