
#ifndef HECATE_EARTH_CANONICALIZER
#define HECATE_EARTH_CANONICALIZER
include "mlir/IR/PatternBase.td"
include "hecate/Dialect/Earth/IR/EarthOps.td"


/* def TestPattern : Pat<(NegateOp(NegateOp $arg)), (replaceWithValue $arg)>; */


def isZero : Constraint<CPred<"$0.dyn_cast<IntegerAttr>().getInt() == 0"> , "check zero">;
def isZeroRotateOffset : Constraint<CPred<"$0.size() == 1 && $0[0] == 0">, "check a single zero rotation offset">;
def isAllZero : Constraint<CPred<"$0.isa<DenseElementsAttr>() && llvm::all_of($0.dyn_cast<DenseElementsAttr>().getValues<double>(),[&](double d){ return d == 0;})">, "check all zero">;
// Short payloads are zero-padded by Runtime, so a prefix of ones is a mask.
// Only a constant covering every physical slot is an identity or negation.
def isAllOne : Constraint<CPred<"$0.isa<DenseElementsAttr>() && $0.dyn_cast<DenseElementsAttr>().getNumElements() == hecate::earth::EarthDialect::polynomialDegree / 2 && llvm::all_of($0.dyn_cast<DenseElementsAttr>().getValues<double>(),[&](double d){ return d == 1.0;})">, "check full-slot all one">;
def isAllMinusOne : Constraint<CPred<"$0.isa<DenseElementsAttr>() && $0.dyn_cast<DenseElementsAttr>().getNumElements() == hecate::earth::EarthDialect::polynomialDegree / 2 && llvm::all_of($0.dyn_cast<DenseElementsAttr>().getValues<double>(),[&](double d){ return d == -1.0;})">, "check full-slot all minus one">;
def geZeroAfterRescale : Constraint<CPred<"$0.dyn_cast<IntegerAttr>().getInt() - hecate::earth::EarthDialect::rescalingFactor >= 0"> , "check zero">;
def singleUser : Constraint<CPred<"$0.hasOneUse()">, "single user">;
def structureNotFixed : Constraint<CPred<"!$0.getDefiningOp()->getAttr(\"sm_fixed\")">, "scale management structure is not fixed">;
def constantScale : NativeCodeCall<"$_builder.getI64IntegerAttr($0.getType().dyn_cast<::hecate::earth::HEScaleTypeInterface>().getScale())">;
// The frontend has no selected runtime profile. Its default ring degree must
// not decide whether a constant covers all slots of the eventual target.
def hasManagedSlotCapacity : Constraint<CPred<"$0.getDefiningOp()->getParentOfType<::mlir::func::FuncOp>()->hasAttr(\"init_level\")">, "require parameter-managed function">;

def ZeroUpscalePattern : Pat<(UpscaleOp $arg, $val), (replaceWithValue $arg), [(isZero $val)]>;
def ZeroModswitchPattern : Pat<(ModswitchOp $arg, $val), (replaceWithValue $arg), [(isZero $val)]>;
def ZeroRotatePattern : Pat<(RotateOp $arg, $offset), (replaceWithValue $arg), [(isZeroRotateOffset $offset)]>;
def AddZeroPattern : Pat<(AddOp (either $lhs, (ConstantOp $val, $dat2)) ), (replaceWithValue $lhs), [(isAllZero $val)]>;
def MulZeroPattern : Pat<(MulOp (either $lhs, (ConstantOp:$res $val, $dat2)) ), (replaceWithValue $res), [(isAllZero $val)]>;
// A full-slot unit still contributes its encoding scale to multiplication.
def MulOnePattern : Pat<(MulOp (either $lhs, (ConstantOp:$res $val, $dat2)) ), (UpscaleOp $lhs, (constantScale $res)), [(isAllOne $val), (hasManagedSlotCapacity $res)]>;
def NegMulPattern : Pat<(MulOp (either $lhs, (ConstantOp:$res $val, $dat2)) ), (UpscaleOp (NegateOp $lhs), (constantScale $res)), [(isAllMinusOne $val), (hasManagedSlotCapacity $res)]>;


// This patterns
def addIntegerAttr : NativeCodeCall<"$_builder.getI64IntegerAttr($0.dyn_cast<IntegerAttr>().getInt() + $1.dyn_cast<IntegerAttr>().getInt())">;
def UpscaleUpscalePattern : Pat<(UpscaleOp (UpscaleOp:$res $arg, $val1), $val2), (UpscaleOp $arg, (addIntegerAttr $val1, $val2)), [(structureNotFixed $res)]>;
def ModswitchModswitchPattern : Pat<(ModswitchOp (ModswitchOp:$res $arg, $val1), $val2), (ModswitchOp $arg, (addIntegerAttr $val1, $val2)), [(structureNotFixed $res)]>;

def changeConstScale : NativeCodeCall<"$_builder.create<::hecate::earth::ConstantOp>($_loc, $0.getType().dyn_cast<::hecate::earth::HEScaleTypeInterface>().switchScale($0.getType().dyn_cast<::hecate::earth::HEScaleTypeInterface>().getScale() + $1.dyn_cast<IntegerAttr>().getInt()), $2, $3)">;
def UpscaleConstantPattern: Pat<(UpscaleOp (ConstantOp:$res $dat1,$dat2), $up), (changeConstScale $res, $up, $dat1, $dat2)>;

// Absorb scale normalization into the plaintext's encoding. This removes the
// later multiplication by an encoded full-slot unit without changing payload,
// output scale, or levels. A shared product must retain its original scale.
def UpscaleMulConstantPattern : Pat<
    (UpscaleOp (MulOp:$product (either $lhs, (ConstantOp:$plain $payload, $rms))), $up),
    (MulOp $lhs, (changeConstScale $plain, $up, $payload, $rms)),
    [(singleUser $product), (structureNotFixed $product)]>;

def changeConstLevel : NativeCodeCall<"$_builder.create<::hecate::earth::ConstantOp>($_loc, $0.getType().dyn_cast<::hecate::earth::HEScaleTypeInterface>().switchLevel($0.getType().dyn_cast<::hecate::earth::HEScaleTypeInterface>().getLevel() + $1.dyn_cast<IntegerAttr>().getInt()), $2, $3)">;
def ModswitchConstantPattern: Pat<(ModswitchOp (ConstantOp:$res $dat1,$dat2), $down), (changeConstLevel $res, $down, $dat1, $dat2)>;

/* def moduloSlot : NativeCodeCall<"$0 % (hecate::earth::EarthDialect::polynomialDegree/2)">; */
/* def RotateOffsetModuloPattern: Pat<(RotateOp $arg, $offset), (RotateOp $arg, (moduloSlot $offset))>; */


def UpscaleRescalePattern: Pat<(UpscaleOp (RescaleOp:$res $arg), $up), (RescaleOp (UpscaleOp $arg, $up)), [(singleUser $res)]>;

def getScaleAfterRescale : NativeCodeCall<"$0.dyn_cast<IntegerAttr>().getInt() - hecate::earth::EarthDialect::rescalingFactor">;

def RescaleUpscalePattern: Pat<(RescaleOp (UpscaleOp $arg, $up) ), (ModswitchOp (UpscaleOp $arg, (getScaleAfterRescale $up) ), (NativeCodeCall<"$_builder.getI64IntegerAttr(1)">) ), [(geZeroAfterRescale $up)]>;

#endif
