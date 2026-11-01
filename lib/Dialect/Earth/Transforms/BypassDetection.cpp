#include "hecate/Dialect/Earth/Analysis/CandidateAnalysis.h"
#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "hecate/Dialect/Earth/Transforms/RegionalScalePlanning.h"
#include "hecate/Support/Support.h"
#include "hecate/Support/CompilationProgress.h"
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
    hecate::CompilationProgress progress("bypass-analysis");
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
    if (hecate::earth::usesRegionalScalePlanning())
      hecate::earth::preserveRegionalOutputs(func);
    if (hecate::earth::usesRegionalScalePlanning() &&
        failed(hecate::earth::scheduleRegionalRegions(func)))
      return signalPassFailure();
    auto &ca = getAnalysis<hecate::CandidateAnalysis>();
    if (hecate::earth::usesRegionalScalePlanning()) {
      auto boundaries = hecate::earth::regionalScaleBoundaries(func);
      if (failed(boundaries))
        return signalPassFailure();
      ca.restrictToRegionalBoundaries(*boundaries);
      llvm::errs() << "Regional DaCapo indexed values: " << ca.getNumValues()
                   << ", boundaries: " << ca.getNumEdges() << '\n';
      hecate::earth::RegionalScaleAnalysis analysis(func, waterline, scale);
      if (hecate::compilationProgressEnabled()) {
        SmallVector<hecate::earth::RegionalScaleState> initial;
        for (Value arg : func.getArguments())
          initial.push_back({arg, static_cast<unsigned>(
              hecate::earth::EarthDialect::nativeLevelAnchor + 1 -
              hecate::earth::getScaleType(arg).getLevel()), static_cast<double>(waterline)});
        auto limit = analysis.coverage(0, initial);
        if (failed(limit))
          return signalPassFailure();
        const auto edges = ca.getEdges();
        llvm::errs() << "[compile] regional-initial-coverage limit=" << *limit
                     << " first_cut=" << (edges.size() > 1 ? edges[1] : -1)
                     << " inputs=" << initial.size() << '\n';
      }
      const unsigned q = hecate::earth::EarthDialect::nativeOutputLevel + 1;
      const unsigned minimumQ = hecate::earth::EarthDialect::nativeLevelAnchor + 1 -
          static_cast<int64_t>(hecate::earth::bootstrapQBudget() * threshold);
      unsigned completed = 0;
      for (auto from : ca.getEdges()) {
        SmallVector<hecate::earth::RegionalScaleState> inputs;
        for (Value arg : func.getArguments())
          inputs.push_back({arg, q, static_cast<double>(scale)});
        if (from)
          for (auto id : ca.getValueInfo(from)->getLiveOuts())
            inputs.push_back({ca.getValueInfo(id)->getValue(), q,
                              static_cast<double>(scale)});
        auto limit = analysis.coverage(from, inputs, minimumQ);
        if (failed(limit))
          return signalPassFailure();
        if (*limit >= 0) {
          ca.getValueInfo(from)->setThresholdOpid(*limit);
          if (!from)
            for (auto input : ca.getValueInfo(0)->getLiveOuts())
              ca.getValueInfo(input)->setThresholdOpid(*limit);
        }
        progress.update(++completed, ca.getNumEdges());
      }
      markAnalysesPreserved<hecate::CandidateAnalysis>();
      return;
    }
    std::atomic<bool> failedAny{false};
    std::vector<std::thread> threads;
    auto maxThreads = func.getContext()->isMultithreadingEnabled()
                          ? std::max(1u, std::thread::hardware_concurrency())
                          : 1u;
    unsigned completed = 0;
    for (auto from : ca.getEdges()) {
      progress.update(completed++, ca.getNumEdges());
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
    OpBuilder builder(func);
    if (hecate::earth::usesRegionalScalePlanning()) {
      // Probe the suffix from a refreshed frontier. The prefix has no known
      // states here and must not be evaluated with threshold scale rules.
      SmallVector<Value> inputs;
      if (from)
        for (auto id : ca.getValueInfo(from)->getLiveOuts())
          inputs.push_back(ca.getValueInfo(id)->getValue());
      auto copy = hecate::earth::cloneRegionalSegment(
          func, inputs, func.front().getTerminator()->getOperands(), {}, from,
          bootstrap_scale, false);
      if (!copy)
        return failure();
      module->push_back(copy);
      const int64_t scale = bootstrap_scale
                                ? bootstrap_scale.getValue()
                                : hecate::earth::EarthDialect::rescalingFactor;
      for (auto arg : copy.getArguments())
        arg.setType(hecate::earth::getScaleType(arg)
                        .switchLevel(hecate::earth::bootstrapOutputLevel())
                        .switchScale(scale));
      copy->setAttr("earth.arg_scales",
                    builder.getDenseF64ArrayAttr(
                        SmallVector<double>(copy.getNumArguments(), scale)));
      hecate::earth::clearActualScales(copy);
      auto minimumQ =
          hecate::earth::EarthDialect::nativeLevelAnchor + 1 -
          static_cast<int64_t>(hecate::earth::bootstrapQBudget() * threshold);
      auto limit = hecate::earth::analyzeRegionalScaleCoverage(copy, waterline,
                                                               scale, minimumQ);
      if (failed(limit))
        return failure();
      if (*limit >= 0) {
        ca.getValueInfo(from)->setThresholdOpid(*limit);
        if (!from)
          for (auto input : ca.getValueInfo(0)->getLiveOuts())
            ca.getValueInfo(input)->setThresholdOpid(*limit);
      }
      return success();
    }
    auto copy = func.clone();
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
    if (hecate::earth::usesNativeScalePlanning()) {
      copy->setAttr("earth.arg_scales", builder.getDenseF64ArrayAttr(
          SmallVector<double>(copy.getNumArguments(), static_cast<double>(waterline))));
      hecate::earth::clearActualScales(copy);
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
      if (!isa<hecate::earth::BootstrapOp>(op) &&
          (id < from || (hecate::earth::usesNativeScalePlanning() && from && id == from)))
        continue;
      // This abstract suffix starts at a refreshed frontier; its raw prefix
      // has no actual input states. Real candidate replay checks Boot inputs.
      auto result = hecate::earth::applyThresholdScale(op, waterline, scale, false);
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
