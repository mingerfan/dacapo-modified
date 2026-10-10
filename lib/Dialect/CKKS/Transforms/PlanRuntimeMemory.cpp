#include "hecate/Dialect/CKKS/Transforms/Passes.h"
#include "hecate/Dialect/CKKS/Transforms/RuntimePlanUtils.h"
#include "hecate/Dialect/Dist/IR/DistOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseSet.h"

namespace hecate::ckks {
#define GEN_PASS_DEF_PLANRUNTIMEMEMORY
#include "hecate/Dialect/CKKS/Transforms/Passes.h.inc"
} // namespace hecate::ckks

using namespace mlir;
namespace ckks = hecate::ckks;
namespace dist = hecate::dist;

namespace {
struct PlanRuntimeMemoryPass
    : ckks::impl::PlanRuntimeMemoryBase<PlanRuntimeMemoryPass> {
  using Base::Base;
  void runOnOperation() override {
    auto func = getOperation();
    if (func->hasAttr("runtime.memory_planned")) {
      func.emitError("runtime memory has already been planned");
      return signalPassFailure();
    }
    auto layout = ckks::buildRuntimePlanLayout(func);
    if (failed(layout))
      return signalPassFailure();
    for (Operation &operation : func.getBody().front())
      if (isa<dist::ReleaseOp>(operation) ||
          operation.hasAttr("runtime.reuse_input")) {
        operation.emitError(
            "memory planning requires IR without Release/reuse annotations");
        return signalPassFailure();
      }
    auto terminator =
        dyn_cast<func::ReturnOp>(func.getBody().front().getTerminator());
    if (!terminator) {
      func.emitError("runtime memory planning requires func.return");
      return signalPassFailure();
    }
    llvm::DenseSet<Value> defined, returned, overwritten;
    for (Value argument : func.getArguments())
      defined.insert(argument);
    for (Value output : terminator.getOperands())
      returned.insert(output);
    llvm::DenseMap<Value, Operation *> lastUse;
    for (const auto &instruction : layout->instructions) {
      Operation *operation = instruction.operation;
      for (Value input : operation->getOperands()) {
        if (!defined.count(input)) {
          operation->emitError("initialization/execution order uses a value "
                               "before its definition");
          return signalPassFailure();
        }
        lastUse[input] = operation;
      }
      for (Value result : operation->getResults()) {
        defined.insert(result);
        lastUse[result] = operation;
      }
      if (enableReuse && ckks::canReuseRuntimeInput(operation, func) &&
          !returned.count(operation->getOperand(0)))
        overwritten.insert(operation->getOperand(0));
    }
    OpBuilder builder(func.getContext());
    // Preserve the exporter's IDs while canonicalizing the two phases in IR.
    for (const auto &[id, value] : layout->values)
      if (auto *producer = value.getDefiningOp())
        producer->setAttr("runtime.value_id", builder.getI64IntegerAttr(id));
    for (const auto &instruction : layout->instructions)
      instruction.operation->moveBefore(terminator);
    for (const auto &instruction : layout->instructions) {
      Operation *operation = instruction.operation;
      if (operation->getNumOperands() &&
          overwritten.count(operation->getOperand(0)) &&
          ckks::canReuseRuntimeInput(operation, func))
        operation->setAttr("runtime.reuse_input", builder.getI64IntegerAttr(0));
      builder.setInsertionPointAfter(operation);
      for (const auto &[id, value] : layout->values) {
        if (returned.count(value) || overwritten.count(value) ||
            lastUse.lookup(value) != operation)
          continue;
        OperationState state(operation->getLoc(),
                             dist::ReleaseOp::getOperationName());
        state.addOperands(value);
        state.addAttribute("initialization",
                           builder.getBoolAttr(instruction.initialization));
        builder.create(state);
      }
    }
    func->setAttr("runtime.memory_planned", builder.getUnitAttr());
  }
};
} // namespace
