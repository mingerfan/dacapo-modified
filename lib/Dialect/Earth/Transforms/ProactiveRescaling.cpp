
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/Debug.h"

#include "hecate/Dialect/Earth/Analysis/ScaleManagementUnit.h"
#include "hecate/Dialect/Earth/Transforms/Common.h"

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_PROACTIVERESCALING
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct ProactiveRescalingPass
    : public hecate::earth::impl::ProactiveRescalingBase<
          ProactiveRescalingPass> {
  ProactiveRescalingPass() {}

  ProactiveRescalingPass(hecate::earth::ProactiveRescalingOptions ops) {
    this->waterline = ops.waterline;
    this->bootstrap_scale = ops.bootstrap_scale;
    this->output_val = ops.output_val;
  }

  void runOnOperation() override {

    auto func = getOperation();

    if (hecate::earth::usesTargetScalePlanning()) {
      func.emitError("experimental target scale planning currently requires greedy placement");
      return signalPassFailure();
    }

    markAnalysesPreserved<hecate::ScaleManagementUnit>();

    mlir::OpBuilder builder(func);
    mlir::IRRewriter rewriter(builder);
    SmallVector<mlir::Type, 4> inputTypes;

    hecate::earth::refineInputValues(func, builder, inputTypes, waterline,
                                     output_val);
    const int64_t bootScale =
        bootstrap_scale ? bootstrap_scale.getValue()
                        : hecate::earth::EarthDialect::rescalingFactor;
    if (waterline <= 0 || hecate::earth::EarthDialect::rescalingFactor <= 0 ||
        bootScale < waterline || hecate::earth::bootstrapQBudget() < 0) {
      func.emitError("invalid threshold scale policy or Boot Q budget");
      return signalPassFailure();
    }
    for (Value input : func.getArguments())
      if (hecate::earth::getScaleType(input).isCipher() &&
          hecate::earth::getScaleType(input).getLevel() >
              hecate::earth::bootstrapQBudget()) {
        func.emitError("input ciphertext exceeds the Boot Q budget");
        return signalPassFailure();
      }
    SmallVector<hecate::earth::ForwardMgmtInterface> operations;
    func.walk([&](hecate::earth::ForwardMgmtInterface op) {
      operations.push_back(op);
    });
    for (auto op : operations) {
      auto result =
          hecate::earth::applyThresholdScale(op, waterline, bootScale);
      if (failed(result)) {
        op->emitError(
            "type inference failed during threshold scale management");
        return signalPassFailure();
      }
      auto type = hecate::earth::getScaleType(*result);
      if (type.isCipher() &&
          type.getLevel() > hecate::earth::bootstrapQBudget()) {
        op->emitError(
            "operation exceeds the Boot Q budget after threshold Rescale");
        return signalPassFailure();
      }
    }
    hecate::earth::finishThresholdScale(func, builder, inputTypes);
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
