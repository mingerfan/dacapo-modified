
#include "hecate/Dialect/Earth/Analysis/ScaleManagementUnit.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "hecate/Dialect/Earth/Analysis/CandidateAnalysis.h"
#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <random>

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_COVERAGERECORDER
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

#define DEBUG_TYPE "dacapo"

namespace {
/// Pass to bufferize Arith ops.
struct CoverageRecorderPass
    : public hecate::earth::impl::CoverageRecorderBase<CoverageRecorderPass> {
  CoverageRecorderPass() {}
  CoverageRecorderPass(hecate::earth::CoverageRecorderOptions ops) {
    this->waterline = ops.waterline;
    this->bootstrap_scale = ops.bootstrap_scale;
    this->threshold = ops.threshold;
  }

  void runOnOperation() override {

    auto func = getOperation();
    auto dup = func.clone();
    auto &&from = dup->getAttrOfType<mlir::DenseI64ArrayAttr>("cutted_edge")
                      .asArrayRef()
                      .front();

    markAnalysesPreserved<hecate::CandidateAnalysis>();
    auto &&block = dup.getBody().front();
    auto &&operations = block.getOperations();
    mlir::OpBuilder builder(dup);

    // Set function argument types
    auto &&inputType_attrs = dup->getAttr("segment_inputType")
                                 .dyn_cast<mlir::ArrayAttr>()
                                 .getValue();
    for (size_t i = 0; i < dup.getNumArguments(); i++) {
      auto argval = dup.getArgument(i);
      auto input_type = inputType_attrs[i]
                            .dyn_cast<mlir::TypeAttr>()
                            .getValue()
                            .dyn_cast<hecate::earth::HEScaleTypeInterface>();
      argval.setType(input_type);
    }
    if (hecate::earth::usesNativeScalePlanning()) {
      hecate::earth::clearActualScales(dup);
      dup->setAttr("earth.arg_scales", dup->getAttr("segment_input_scales"));
    }

    // Analyze normalized results, not the pre-Rescale multiply scale.
    int64_t coverage = -1, bootCoverage = -1;
    SmallVector<hecate::earth::ForwardMgmtInterface> original;
    for (auto &op : operations)
      if (auto forward = dyn_cast<hecate::earth::ForwardMgmtInterface>(op))
        original.push_back(forward);
    const int64_t bootScale =
        bootstrap_scale ? bootstrap_scale.getValue()
                        : hecate::earth::EarthDialect::rescalingFactor;
    for (auto sop : original) {
      auto opid = hecate::getIntegerAttr("opid", sop->getResult(0));
      if (!isa<hecate::earth::BootstrapOp>(sop) &&
          (opid < from || (hecate::earth::usesNativeScalePlanning() && from && opid == from)))
        continue;
      auto result =
          hecate::earth::applyThresholdScale(sop, waterline, bootScale);
      if (failed(result)) {
        sop->emitError("type inference failed during Boot coverage analysis");
        dup.erase();
        return signalPassFailure();
      }
      auto type = hecate::earth::getScaleType(*result);
      if (type.isCipher() &&
          type.getLevel() > hecate::earth::bootstrapQBudget()) {
        coverage = bootCoverage = opid;
        break;
      }
    }
    dup.erase();
    func->setAttr("coverages",
                  builder.getDenseI64ArrayAttr({coverage, bootCoverage}));
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
