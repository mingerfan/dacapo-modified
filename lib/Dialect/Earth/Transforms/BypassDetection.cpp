#include "hecate/Dialect/Earth/Analysis/CandidateAnalysis.h"
#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "hecate/Support/Support.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include <atomic>
#include <thread>

namespace hecate::earth {
#define GEN_PASS_DEF_BYPASSDETECTION
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace hecate::earth
using namespace mlir;
namespace {
struct BypassDetectionPass
    : hecate::earth::impl::BypassDetectionBase<BypassDetectionPass> {
  BypassDetectionPass() = default;
  BypassDetectionPass(hecate::earth::BypassDetectionOptions options) {
    waterline = options.waterline;
    threshold = options.threshold;
    bootstrap_scale = options.bootstrap_scale;
  }
  void runOnOperation() override {
    auto func = getOperation();
    if (func.isExternal())
      return;
    if (!llvm::hasSingleElement(func.getBody())) {
      func.emitError("DaCapo Boot analysis requires a straight-line function");
      return signalPassFailure();
    }
    auto scale = bootstrap_scale ? bootstrap_scale.getValue()
                                 : hecate::earth::EarthDialect::rescalingFactor;
    if (waterline <= 0 || hecate::earth::EarthDialect::rescalingFactor <= 0 ||
        scale < waterline || hecate::earth::bootstrapQBudget() < 0) {
      func.emitError("invalid threshold scale policy or Boot Q budget");
      return signalPassFailure();
    }
    for (auto argument : func.getArguments()) {
      auto type = hecate::earth::getScaleType(argument);
      if (!type || !type.isCipher() ||
          type.getLevel() > hecate::earth::bootstrapQBudget()) {
        func.emitError(
            "DaCapo Boot analysis requires bootstrappable ciphertext inputs");
        return signalPassFailure();
      }
    }
    for (auto &op : llvm::make_early_inc_range(llvm::reverse(func.front())))
      if (isOpTriviallyDead(&op))
        op.erase();
    for (auto &op : func.front())
      if (!isa<hecate::earth::ConstantOp, hecate::earth::MulOp,
               hecate::earth::AddOp, hecate::earth::RotateOp,
               hecate::earth::NegateOp, func::ReturnOp>(op)) {
        op.emitError("unsupported operation in DaCapo Boot analysis");
        return signalPassFailure();
      }
    auto &ca = getAnalysis<hecate::CandidateAnalysis>();
    std::atomic<bool> failedAny{false};
    std::vector<std::thread> threads;
    auto maxThreads = func.getContext()->isMultithreadingEnabled()
                          ? std::max(1u, std::thread::hardware_concurrency())
                          : 1u;
    for (auto from : ca.getEdges()) {
      threads.emplace_back([&, from] {
        if (failed(findBypassEdge(func, ca, from)))
          failedAny = true;
      });
      if (threads.size() >= maxThreads) {
        for (auto &thread : threads)
          thread.join();
        threads.clear();
      }
    }
    for (auto &thread : threads)
      thread.join();
    if (failedAny)
      return signalPassFailure();
    markAnalysesPreserved<hecate::CandidateAnalysis>();
  }
  LogicalResult findBypassEdge(func::FuncOp func, hecate::CandidateAnalysis &ca,
                               int64_t from) {
    OwningOpRef<ModuleOp> module = ModuleOp::create(func.getLoc());
    auto copy = func.clone();
    OpBuilder builder(copy);
    copy->setAttr("btp_at", builder.getI64IntegerAttr(from));
    copy->setAttr("btp_target", builder.getDenseI64ArrayAttr(
                                    ca.getValueInfo(from)->getLiveOuts()));
    module->push_back(copy);
    PassManager pm(func.getContext());
    pm.addNestedPass<func::FuncOp>(
        hecate::earth::createBootstrapPlacement({bootstrap_scale}));
    if (failed(pm.run(*module)))
      return failure();
    for (auto argument : copy.getArguments()) {
      auto type = hecate::earth::getScaleType(argument);
      argument.setType(type.switchScale(waterline));
    }
    SmallVector<hecate::earth::ForwardMgmtInterface> operations;
    copy.walk([&](hecate::earth::ForwardMgmtInterface op) {
      operations.push_back(op);
    });
    const int64_t scale = bootstrap_scale
                              ? bootstrap_scale.getValue()
                              : hecate::earth::EarthDialect::rescalingFactor;
    for (auto op : operations) {
      auto id = hecate::getIntegerAttr("opid", op->getResult(0));
      if (!isa<hecate::earth::BootstrapOp>(op) && id < from)
        continue;
      auto result = hecate::earth::applyThresholdScale(op, waterline, scale);
      if (failed(result)) {
        op->emitError("type inference failed during Boot bypass analysis");
        return failure();
      }
      auto type = hecate::earth::getScaleType(*result);
      if (type.isCipher() &&
          type.getLevel() > hecate::earth::bootstrapQBudget() * threshold) {
        ca.getValueInfo(from)->setThresholdOpid(id);
        if (!from)
          for (auto input : ca.getValueInfo(0)->getLiveOuts())
            ca.getValueInfo(input)->setThresholdOpid(id);
        break;
      }
    }
    return success();
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
