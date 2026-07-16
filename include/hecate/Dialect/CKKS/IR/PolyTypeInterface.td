#ifndef HECATE_CKKS_POLY_TYPE_INTERFACE
#define HECATE_CKKS_POLY_TYPE_INTERFACE

include "mlir/IR/OpBase.td"

def PolyTypeInterface : TypeInterface<"PolyTypeInterface"> {
  let description = [{
    Accesses the CKKS components, binary scale exponent and modulus-chain
    level carried by a plaintext or ciphertext type.
  }];
  let cppNamespace = "::hecate::ckks";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{Get the number of polynomial components.}],
      /*retType=*/"unsigned",
      /*methodName=*/"getComponents"
    >,
    InterfaceMethod<
      /*desc=*/[{Get the binary scale exponent.}],
      /*retType=*/"unsigned",
      /*methodName=*/"getScaleLog2"
    >,
    InterfaceMethod<
      /*desc=*/[{Get the RuntimePlan-direction modulus-chain level.}],
      /*retType=*/"unsigned",
      /*methodName=*/"getLevel"
    >,
    InterfaceMethod<
      /*desc=*/[{Return the same type at another level.}],
      /*retType=*/"::hecate::ckks::PolyTypeInterface",
      /*methodName=*/"switchLevel",
      /*args=*/(ins "unsigned":$level)
    >,
    InterfaceMethod<
      /*desc=*/[{Return the same type with another component count.}],
      /*retType=*/"::hecate::ckks::PolyTypeInterface",
      /*methodName=*/"switchComponents",
      /*args=*/(ins "unsigned":$components)
    >,
    InterfaceMethod<
      /*desc=*/[{Return the same type with another binary scale exponent.}],
      /*retType=*/"::hecate::ckks::PolyTypeInterface",
      /*methodName=*/"switchScaleLog2",
      /*args=*/(ins "unsigned":$scale_log2)
    >
  ];
}

#endif
