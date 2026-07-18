
#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cstdint>

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

::mlir::LogicalResult hecate::ckks::EncodeOp::verify() {
  if (ckks::getPolyType(getResult()).getComponents() != 1)
    return emitOpError("result must be plaintext with one component");
  return ::mlir::success();
}

::mlir::LogicalResult hecate::ckks::RotateCOp::verify() {
  if (ckks::getPolyType(getSrc()).getComponents() != 2)
    return emitOpError("source must be a two-component ciphertext");
  if (getOffset().size() != 1 || getOffset()[0] == 0)
    return emitOpError("requires exactly one nonzero rotation step");
  return ::mlir::success();
}

::mlir::LogicalResult hecate::ckks::NegateCOp::verify() {
  if (ckks::getPolyType(getSrc()).getComponents() < 2)
    return emitOpError("source must be ciphertext");
  return ::mlir::success();
}

::mlir::LogicalResult hecate::ckks::RelinearizeOp::verify() {
  auto result = ckks::getPolyType(getResult());
  auto src = ckks::getPolyType(getSrc());
  if (src.getComponents() != 3 || result.getComponents() != 2 ||
      result.getLevel() != src.getLevel() ||
      result.getScaleLog2() != src.getScaleLog2())
    return emitOpError(
        "requires components 3 -> 2 with unchanged scale and level");
  return ::mlir::success();
}

::mlir::LogicalResult hecate::ckks::RescaleCOp::verify() {
  auto result = ckks::getPolyType(getResult());
  auto src = ckks::getPolyType(getSrc());
  const bool validLevel = result.getLevel() < src.getLevel();
  if (src.getComponents() < 2 ||
      result.getComponents() != src.getComponents() || !validLevel ||
      result.getScaleLog2() >= src.getScaleLog2())
    return emitOpError("requires ciphertext components to remain unchanged and "
                       "level and scale to decrease");
  return ::mlir::success();
}

::mlir::LogicalResult hecate::ckks::ModswitchCOp::verify() {
  auto result = ckks::getPolyType(getResult());
  auto src = ckks::getPolyType(getSrc());
  if (getDownFactor() <= 0)
    return emitOpError("downFactor must be positive");
  const uint64_t downFactor = static_cast<uint64_t>(getDownFactor());
  const bool validLevel = result.getLevel() < src.getLevel() &&
                          (result.getLevel() == 0 ||
                           (downFactor <= src.getLevel() &&
                            result.getLevel() == src.getLevel() - downFactor));
  if (src.getComponents() < 2 ||
      result.getComponents() != src.getComponents() ||
      result.getScaleLog2() != src.getScaleLog2() || !validLevel)
    return emitOpError("requires unchanged ciphertext components and scale "
                       "with the requested lower level");
  return ::mlir::success();
}

::mlir::LogicalResult hecate::ckks::UpscaleCOp::verify() {
  auto result = ckks::getPolyType(getResult());
  auto src = ckks::getPolyType(getSrc());
  if (getUpFactor() <= 0)
    return emitOpError("upFactor must be positive");
  const uint64_t expectedScale =
      static_cast<uint64_t>(src.getScaleLog2()) + getUpFactor();
  if (src.getComponents() < 2 ||
      result.getComponents() != src.getComponents() ||
      result.getLevel() != src.getLevel() ||
      result.getScaleLog2() != expectedScale)
    return emitOpError("requires unchanged ciphertext components and level "
                       "with the requested higher scale");
  return ::mlir::success();
}

::mlir::LogicalResult hecate::ckks::BootstrapCOp::verify() {
  auto result = ckks::getPolyType(getResult());
  auto src = ckks::getPolyType(getSrc());
  if (src.getComponents() < 2 || result.getComponents() != src.getComponents())
    return emitOpError("requires ciphertext input and unchanged components");
  return ::mlir::success();
}

::mlir::LogicalResult hecate::ckks::AddCCOp::verify() {
  if (ckks::getPolyType(getLhs()).getComponents() < 2)
    return emitOpError("operands and result must be ciphertext");
  return ::mlir::success();
}

::mlir::LogicalResult hecate::ckks::AddCPOp::verify() {
  auto result = ckks::getPolyType(getResult());
  auto lhs = ckks::getPolyType(getLhs());
  auto rhs = ckks::getPolyType(getRhs());
  if (lhs.getComponents() < 2 || rhs.getComponents() != 1 ||
      lhs.getLevel() != rhs.getLevel() ||
      lhs.getScaleLog2() != rhs.getScaleLog2() ||
      result.getComponents() != lhs.getComponents() ||
      result.getLevel() != lhs.getLevel() ||
      result.getScaleLog2() != lhs.getScaleLog2())
    return emitOpError("requires ciphertext lhs, plaintext rhs, and unchanged "
                       "output metadata");
  return ::mlir::success();
}

::mlir::LogicalResult hecate::ckks::MulCCOp::verify() {
  auto result = ckks::getPolyType(getResult());
  auto lhs = ckks::getPolyType(getLhs());
  auto rhs = ckks::getPolyType(getRhs());
  if (lhs.getComponents() < 2 || rhs.getComponents() < 2)
    return emitOpError("operands must be ciphertext");
  const uint64_t expectedComponents =
      static_cast<uint64_t>(lhs.getComponents()) + rhs.getComponents() - 1;
  const uint64_t expectedScale =
      static_cast<uint64_t>(lhs.getScaleLog2()) + rhs.getScaleLog2();
  if (lhs.getLevel() != rhs.getLevel() ||
      result.getComponents() != expectedComponents ||
      result.getLevel() != lhs.getLevel() ||
      result.getScaleLog2() != expectedScale)
    return emitOpError("requires ciphertext operands with matching levels and "
                       "multiplied output metadata");
  return ::mlir::success();
}

::mlir::LogicalResult hecate::ckks::MulCPOp::verify() {
  auto result = ckks::getPolyType(getResult());
  auto lhs = ckks::getPolyType(getLhs());
  auto rhs = ckks::getPolyType(getRhs());
  const uint64_t expectedScale =
      static_cast<uint64_t>(lhs.getScaleLog2()) + rhs.getScaleLog2();
  if (lhs.getComponents() < 2 || rhs.getComponents() != 1 ||
      lhs.getLevel() != rhs.getLevel() ||
      result.getComponents() != lhs.getComponents() ||
      result.getLevel() != lhs.getLevel() ||
      result.getScaleLog2() != expectedScale)
    return emitOpError("requires ciphertext lhs, plaintext rhs, and multiplied "
                       "output metadata");
  return ::mlir::success();
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
