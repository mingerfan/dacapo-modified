
#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

#include "hecate/Dialect/CKKS/IR/PolyTypeInterface.h"

#include "hecate/Dialect/CKKS/IR/PolyTypeInterfaceTypes.cpp.inc"

#include "hecate/Dialect/CKKS/IR/PolyTypeInterface.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "hecate/Dialect/CKKS/IR/CKKSOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "hecate/Dialect/CKKS/IR/CKKSOps.cpp.inc"

#include "hecate/Dialect/CKKS/IR/CKKSOpsDialect.cpp.inc"
#include "hecate/Dialect/Earth/IR/EarthOps.h"

/* #include "hecate/Dialect/CKKS/IR/EarthCanonicalizerPattern.inc" */

struct PolyTypeTensorModel
    : public hecate::ckks::PolyTypeInterface::ExternalModel<PolyTypeTensorModel,
                                                            RankedTensorType> {
  unsigned getComponents(Type t) const {
    if (auto polyType = t.dyn_cast<mlir::RankedTensorType>()
                            .getElementType()
                            .dyn_cast<hecate::ckks::PolyTypeInterface>()) {
      return polyType.getComponents();
    } else {
      return 0;
    }
  }
  unsigned getScaleLog2(Type t) const {
    if (auto polyType = t.dyn_cast<mlir::RankedTensorType>()
                            .getElementType()
                            .dyn_cast<hecate::ckks::PolyTypeInterface>()) {
      return polyType.getScaleLog2();
    } else {
      return 0;
    }
  }
  unsigned getLevel(Type t) const {
    if (auto polyType = t.dyn_cast<mlir::RankedTensorType>()
                            .getElementType()
                            .dyn_cast<hecate::ckks::PolyTypeInterface>()) {
      return polyType.getLevel();
    } else {
      return 0;
    }
  }

  hecate::ckks::PolyTypeInterface switchLevel(Type t, unsigned level) const {
    return dyn_cast<hecate::ckks::PolyTypeInterface>(
        mlir::RankedTensorType::get(
            t.dyn_cast<mlir::RankedTensorType>().getShape(),
            t.dyn_cast<mlir::RankedTensorType>()
                .getElementType()
                .dyn_cast<hecate::ckks::PolyTypeInterface>()
                .switchLevel(level)));
  }
  hecate::ckks::PolyTypeInterface switchComponents(Type t,
                                                   unsigned components) const {
    return dyn_cast<hecate::ckks::PolyTypeInterface>(
        mlir::RankedTensorType::get(
            t.dyn_cast<mlir::RankedTensorType>().getShape(),
            t.dyn_cast<mlir::RankedTensorType>()
                .getElementType()
                .dyn_cast<hecate::ckks::PolyTypeInterface>()
                .switchComponents(components)));
  }
  hecate::ckks::PolyTypeInterface switchScaleLog2(Type t,
                                                  unsigned scaleLog2) const {
    return dyn_cast<hecate::ckks::PolyTypeInterface>(
        mlir::RankedTensorType::get(
            t.dyn_cast<mlir::RankedTensorType>().getShape(),
            t.dyn_cast<mlir::RankedTensorType>()
                .getElementType()
                .dyn_cast<hecate::ckks::PolyTypeInterface>()
                .switchScaleLog2(scaleLog2)));
  }
};

void hecate::ckks::CKKSDialect::initialize() {
  // Registers all the Types into the EVADialect class
  addTypes<
#define GET_TYPEDEF_LIST
#include "hecate/Dialect/CKKS/IR/CKKSOpsTypes.cpp.inc"
      >();

  // Registers all the Operations into the EVADialect class
  addOperations<
#define GET_OP_LIST
#include "hecate/Dialect/CKKS/IR/CKKSOps.cpp.inc"
      >();
  mlir::RankedTensorType::attachInterface<PolyTypeTensorModel>(*getContext());
}

::mlir::LogicalResult hecate::ckks::EncodeOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {

  auto op = EncodeOpAdaptor(operands, attributes, properties, regions);
  auto dPoly = ckks::getPolyType(op.getDst());
  if (dPoly.getComponents() == 1) {
    inferredReturnTypes.push_back(op.getDst().getType());
    return ::mlir::success();
  } else {
    return ::mlir::failure();
  }
}

::mlir::LogicalResult hecate::ckks::RescaleCOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = RescaleCOpAdaptor(operands, attributes, properties, regions);
  auto dPoly = ckks::getPolyType(op.getDst());
  auto lPoly = ckks::getPolyType(op.getSrc());
  if (dPoly.getComponents() == lPoly.getComponents() &&
      (dPoly.getLevel() == lPoly.getLevel() - 1 || dPoly.getLevel() == 0)) {
    inferredReturnTypes.push_back(op.getDst().getType());
    return ::mlir::success();
  } else {
    return ::mlir::failure();
  }
}

::mlir::LogicalResult hecate::ckks::ModswitchCOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = ModswitchCOpAdaptor(operands, attributes, properties, regions);
  auto dPoly = ckks::getPolyType(op.getDst());
  auto lPoly = ckks::getPolyType(op.getSrc());
  if (dPoly.getComponents() == lPoly.getComponents() &&
      (dPoly.getLevel() == lPoly.getLevel() - op.getDownFactor() ||
       dPoly.getLevel() == 0)) {
    inferredReturnTypes.push_back(op.getDst().getType());
    return ::mlir::success();
  } else {
    return ::mlir::failure();
  }
}

::mlir::LogicalResult hecate::ckks::UpscaleCOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = UpscaleCOpAdaptor(operands, attributes, properties, regions);
  auto dPoly = ckks::getPolyType(op.getDst());
  auto sPoly = ckks::getPolyType(op.getSrc());
  if (dPoly.getComponents() == sPoly.getComponents() &&
      dPoly.getLevel() == sPoly.getLevel() &&
      dPoly.getScaleLog2() == sPoly.getScaleLog2() + op.getUpFactor()) {
    inferredReturnTypes.push_back(op.getDst().getType());
    return ::mlir::success();
  }
  return ::mlir::failure();
}

::mlir::LogicalResult hecate::ckks::BootstrapCOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = BootstrapCOpAdaptor(operands, attributes, properties, regions);
  auto dPoly = ckks::getPolyType(op.getDst());
  auto lPoly = ckks::getPolyType(op.getSrc());
  if (dPoly.getComponents() == lPoly.getComponents()) {
    inferredReturnTypes.push_back(op.getDst().getType());
    return ::mlir::success();
  } else {
    return ::mlir::failure();
  }
}

::mlir::LogicalResult hecate::ckks::AddCPOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = AddCPOpAdaptor(operands, attributes, properties, regions);
  auto dPoly = ckks::getPolyType(op.getDst());
  auto lPoly = ckks::getPolyType(op.getLhs());
  auto rPoly = ckks::getPolyType(op.getRhs());
  if (std::min(lPoly.getComponents(), rPoly.getComponents()) == 1 &&
      dPoly.getComponents() ==
          std::max(rPoly.getComponents(), lPoly.getComponents()) &&
      lPoly.getLevel() == rPoly.getLevel() &&
      dPoly.getLevel() == lPoly.getLevel()) {
    inferredReturnTypes.push_back(op.getDst().getType());
    return ::mlir::success();
  } else {
    return ::mlir::failure();
  }
}

::mlir::LogicalResult hecate::ckks::MulCPOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = MulCPOpAdaptor(operands, attributes, properties, regions);
  auto dPoly = ckks::getPolyType(op.getDst());
  auto lPoly = ckks::getPolyType(op.getLhs());
  auto rPoly = ckks::getPolyType(op.getRhs());

  if (std::min(lPoly.getComponents(), rPoly.getComponents()) == 1 &&
      dPoly.getComponents() ==
          std::max(lPoly.getComponents(), rPoly.getComponents()) &&
      lPoly.getLevel() == rPoly.getLevel() &&
      lPoly.getLevel() == dPoly.getLevel() &&
      dPoly.getScaleLog2() == lPoly.getScaleLog2() + rPoly.getScaleLog2()) {
    inferredReturnTypes.push_back(op.getDst().getType());
    return ::mlir::success();
  } else {
    return ::mlir::failure();
  }
}

::mlir::LogicalResult hecate::ckks::MulCCOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = MulCCOpAdaptor(operands, attributes, properties, regions);
  auto dPoly = ckks::getPolyType(op.getDst());
  auto lPoly = ckks::getPolyType(op.getLhs());
  auto rPoly = ckks::getPolyType(op.getRhs());

  if (lPoly.getComponents() >= 2 && rPoly.getComponents() >= 2 &&
      dPoly.getComponents() ==
          lPoly.getComponents() + rPoly.getComponents() - 1 &&
      lPoly.getLevel() == rPoly.getLevel() &&
      dPoly.getLevel() == lPoly.getLevel() &&
      dPoly.getScaleLog2() == lPoly.getScaleLog2() + rPoly.getScaleLog2()) {
    inferredReturnTypes.push_back(op.getDst().getType());
    return ::mlir::success();
  }
  return ::mlir::failure();
}

::mlir::LogicalResult hecate::ckks::RelinearizeOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = RelinearizeOpAdaptor(operands, attributes, properties, regions);
  auto dPoly = ckks::getPolyType(op.getDst());
  auto sPoly = ckks::getPolyType(op.getSrc());

  if (sPoly.getComponents() == 3 && dPoly.getComponents() == 2 &&
      dPoly.getLevel() == sPoly.getLevel() &&
      dPoly.getScaleLog2() == sPoly.getScaleLog2()) {
    inferredReturnTypes.push_back(op.getDst().getType());
    return ::mlir::success();
  }
  return ::mlir::failure();
}

hecate::ckks::PolyTypeInterface
hecate::ckks::PolyType::switchLevel(unsigned level) const {
  return get(getContext(), getComponents(), getScaleLog2(), level);
}
hecate::ckks::PolyTypeInterface
hecate::ckks::PolyType::switchComponents(unsigned components) const {
  return get(getContext(), components, getScaleLog2(), getLevel());
}
hecate::ckks::PolyTypeInterface
hecate::ckks::PolyType::switchScaleLog2(unsigned scaleLog2) const {
  return get(getContext(), getComponents(), scaleLog2, getLevel());
}

mlir::RankedTensorType hecate::ckks::getTensorType(mlir::Value v) {
  return v.getType().dyn_cast<mlir::RankedTensorType>();
}
hecate::ckks::PolyTypeInterface hecate::ckks::getPolyType(mlir::Value v) {
  return v.getType().dyn_cast<hecate::ckks::PolyTypeInterface>();
  /* .dyn_cast<mlir::TensorType>() */
  /* .getElementType() */
  /* .dyn_cast<hecate::earth::HEScaleTypeInterface>(); */
}
