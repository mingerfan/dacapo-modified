#include "hecate/Dialect/Earth/Analysis/CandidateAnalysis.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/PassManager.h"
#include <algorithm>

namespace hecate::earth {
#define GEN_PASS_DEF_CANDIDATESELECTION
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace hecate::earth
using namespace mlir;

namespace {
struct CandidateSelectionPass
    : hecate::earth::impl::CandidateSelectionBase<CandidateSelectionPass> {
  CandidateSelectionPass() = default;
  CandidateSelectionPass(hecate::earth::CandidateSelectionOptions options) {
    waterline = options.waterline;
    output_val = options.output_val;
    bootstrap_scale = options.bootstrap_scale;
  }
  void runOnOperation() override {
    auto func = getOperation();
    auto &ca = getAnalysis<hecate::CandidateAnalysis>();
    for (auto cut : ca.getEdges()) {
      SmallVector<int64_t, 4> targets;
      for (auto id : ca.getValueInfo(cut)->getLiveOuts())
        if (!ca.getValueInfo(id)->isBypassEdge(cut))
          targets.push_back(id);
      ca.getValueInfo(cut)->setValidLiveOuts(targets);
      ca.sortValidCandidates(cut);
    }
    OpBuilder builder(func);
    OwningOpRef<ModuleOp> module = ModuleOp::create(func.getLoc());
    PassManager pm(func.getContext());
    pm.addNestedPass<func::FuncOp>(
        hecate::earth::createBootstrapPlacement({bootstrap_scale}));
    pm.addNestedPass<func::FuncOp>(hecate::earth::createProactiveRescaling(
        {waterline, output_val, bootstrap_scale}));
    // Candidate-set zero checks the no-Boot path. Initial refresh is optional
    // and only concerns inputs which have already consumed Q.
    for (size_t size = 0; size <= ca.getMaxNumOuts(); ++size) {
      SmallVector<int64_t> cuts;
      for (size_t group = 1; group <= size; ++group)
        cuts.append(ca.getCandidateSet(group));
      llvm::sort(cuts);
      cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
      for (bool initial : {false, true}) {
        auto initialTargets = ca.getInitialTargets();
        if (initial && initialTargets.empty())
          continue;
        SmallVector<Attribute> entries;
        if (initial) {
          SmallVector<int64_t> entry{0};
          entry.append(initialTargets);
          entries.push_back(builder.getDenseI64ArrayAttr(entry));
        }
        for (auto cut : cuts) {
          if (!cut || cut == ca.getRetOpid())
            continue;
          SmallVector<int64_t> entry{cut};
          entry.append(ca.getTargets(cut, size));
          entries.push_back(builder.getDenseI64ArrayAttr(entry));
        }
        auto copy = func.clone();
        copy->setAttr("btp_schedule", builder.getArrayAttr(entries));
        module->push_back(copy);
        LogicalResult feasible = failure();
        {
          // Infeasible trial plans are normal search outcomes. A failed search
          // emits one diagnostic below rather than errors from rejected trials.
          ScopedDiagnosticHandler diagnostics(
              func.getContext(), [](Diagnostic &) { return success(); });
          feasible = pm.run(*module);
        }
        copy.erase();
        if (succeeded(feasible)) {
          func->setAttr("selected_set", builder.getI64IntegerAttr(size));
          ca.finalizeCandidates(size);
          markAnalysesPreserved<hecate::CandidateAnalysis>();
          return;
        }
      }
    }
    func.emitError(
        "no feasible DaCapo candidate set for the threshold Q budget");
    signalPassFailure();
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
