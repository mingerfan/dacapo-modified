#include "hecate/Dialect/CKKS/Transforms/RuntimePlanUtils.h"
#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "hecate/Dialect/Dist/IR/DistOps.h"
#include <algorithm>
#include <limits>

using namespace mlir;

namespace hecate::ckks {

bool isRuntimeInitialization(Operation *operation) {
  if (isa<EncodeOp>(operation))
    return !operation->hasAttr("runtime.online_encode");
  if (auto transfer = dyn_cast<dist::TransferOp>(operation))
    return transfer.getInitialization();
  if (auto release = dyn_cast<dist::ReleaseOp>(operation))
    return release.getInitialization();
  return false;
}

FailureOr<RuntimePlace> runtimePlace(Value value, func::FuncOp func) {
  IntegerAttr rank, device;
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    rank = func.getArgAttrOfType<IntegerAttr>(argument.getArgNumber(),
                                              "dist.rank");
    device = func.getArgAttrOfType<IntegerAttr>(argument.getArgNumber(),
                                                "dist.device");
  } else {
    rank = value.getDefiningOp()->getAttrOfType<IntegerAttr>("dist.rank");
    device = value.getDefiningOp()->getAttrOfType<IntegerAttr>("dist.device");
  }
  if (!rank || !device) {
    if (!func->hasAttr("dist.device_counts"))
      return RuntimePlace{};
    func.emitError("placed value is missing dist.rank/dist.device");
    return failure();
  }
  if (rank.getInt() < 0 || device.getInt() < -1) {
    func.emitError("invalid physical place");
    return failure();
  }
  if (auto counts =
          func->getAttrOfType<DenseI64ArrayAttr>("dist.device_counts")) {
    if (rank.getInt() >= counts.size() ||
        (device.getInt() >= 0 &&
         device.getInt() >= counts.asArrayRef()[rank.getInt()])) {
      func.emitError("physical place is outside dist.device_counts");
      return failure();
    }
  }
  return RuntimePlace{rank.getInt(), device.getInt()};
}

FailureOr<RuntimePlanLayout> buildRuntimePlanLayout(func::FuncOp func) {
  if (!func.getBody().hasOneBlock()) {
    func.emitError("RuntimePlan requires a single-block function");
    return failure();
  }
  RuntimePlanLayout layout;
  uint64_t implicitId = 0, nextId = 0;
  const auto registerId = [&](Value value, uint64_t id) -> LogicalResult {
    if (id == std::numeric_limits<uint64_t>::max() || layout.ids.count(value) ||
        layout.values.count(id))
      return func.emitError("duplicate or out-of-range RuntimePlan ValueId");
    layout.ids.insert({value, id});
    layout.values.emplace(id, value);
    nextId = std::max(nextId, id + 1);
    return success();
  };
  for (BlockArgument argument : func.getArguments())
    if (failed(registerId(argument, implicitId++)))
      return failure();
  const bool placed = func->hasAttr("dist.device_counts");
  for (Operation &operation : func.getBody().front()) {
    if (operation.getName().getDialectNamespace() != "ckks")
      continue;
    if (operation.getNumResults() != 1 || operation.getNumRegions() != 0) {
      operation.emitError(
          "RuntimePlan expects linear CKKS operations with one result");
      return failure();
    }
    auto logical = operation.getAttrOfType<IntegerAttr>("dist.logical_id");
    auto stable = operation.getAttrOfType<IntegerAttr>("runtime.value_id");
    if (placed && !logical) {
      operation.emitError("placed CKKS operation is missing logical ValueId");
      return failure();
    }
    if ((logical && logical.getInt() < 0) || (stable && stable.getInt() < 0) ||
        (logical && stable && logical.getInt() != stable.getInt())) {
      operation.emitError("invalid or inconsistent RuntimePlan ValueId");
      return failure();
    }
    uint64_t id = stable    ? stable.getInt()
                  : logical ? logical.getInt()
                            : implicitId++;
    if (failed(registerId(operation.getResult(0), id)))
      return failure();
  }
  for (Operation &operation : func.getBody().front()) {
    if (isa<dist::TransferOp, dist::PreparePlainOp>(operation)) {
      auto stable = operation.getAttrOfType<IntegerAttr>("runtime.value_id");
      if (stable && stable.getInt() < 0) {
        operation.emitError("invalid RuntimePlan transfer ValueId");
        return failure();
      }
      if (failed(registerId(operation.getResult(0),
                            stable ? stable.getInt() : nextId)))
        return failure();
    }
  }
  for (bool initialization : {true, false})
    for (Operation &operation : func.getBody().front()) {
      if (isa<func::ReturnOp>(operation))
        continue;
      if (!isa<dist::ReleaseOp, dist::FenceOp>(operation) && operation.getNumResults() != 1) {
        operation.emitError("unsupported RuntimePlan operation");
        return failure();
      }
      if (isRuntimeInitialization(&operation) == initialization)
        layout.instructions.push_back(
            {&operation, initialization, layout.instructions.size()});
    }
  return layout;
}

bool canReuseRuntimeInput(Operation *operation, func::FuncOp func) {
  if (operation->getNumOperands() == 0 || operation->getNumResults() != 1)
    return false;
  Value input = operation->getOperand(0), output = operation->getResult(0);
  Operation *producer = input.getDefiningOp();
  if (!producer || producer->getName().getDialectNamespace() != "ckks" ||
      isa<EncodeOp>(producer) || !input.hasOneUse() ||
      input.getType() != output.getType())
    return false;
  auto source = runtimePlace(input, func),
       destination = runtimePlace(output, func);
  if (failed(source) || failed(destination) || !(*source == *destination))
    return false;
  return destination->device == -1 ? isa<NegateCOp, RotateCOp>(operation)
                                   : isa<AddCPOp, RotateCOp>(operation);
}

} // namespace hecate::ckks
