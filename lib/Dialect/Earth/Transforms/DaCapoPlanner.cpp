#include "hecate/Dialect/Earth/Analysis/CandidateAnalysis.h"
#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <algorithm>
#include <map>

namespace hecate::earth {
#define GEN_PASS_DEF_DACAPOPLANNER
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace hecate::earth
using namespace mlir;
namespace {
struct DaCapoPlannerPass
    : hecate::earth::impl::DaCapoPlannerBase<DaCapoPlannerPass> {
  DaCapoPlannerPass() = default;
  DaCapoPlannerPass(hecate::earth::DaCapoPlannerOptions options) {
    waterline = options.waterline;
    output_val = options.output_val;
    bootstrap_scale = options.bootstrap_scale;
  }
  void runOnOperation() override {
    auto func = getOperation();
    auto &ca = getAnalysis<hecate::CandidateAnalysis>();
    OpBuilder builder(func);
    struct Plan {
      double cost = 0;
      SmallVector<int64_t> cuts;
      SmallVector<Type, 4> types;
    };
    // Bypassed ciphertexts retain scale and consumed Q. Different frontier
    // states must survive the DP even when a cheaper state exists at that cut.
    std::map<int64_t, SmallVector<Plan>> plans;
    SmallVector<Type, 4> inputs;
    for (auto argument : func.getArguments())
      inputs.push_back(
          hecate::earth::getScaleType(argument).switchScale(waterline));
    plans[0].push_back({0, {}, inputs});
    OwningOpRef<ModuleOp> module = ModuleOp::create(func.getLoc());
    PassManager pm(func.getContext()), coverage(func.getContext());
    pm.addNestedPass<func::FuncOp>(
        hecate::earth::createBootstrapPlacement({bootstrap_scale}));
    pm.addNestedPass<func::FuncOp>(hecate::earth::createCodeSegmentation());
    pm.addNestedPass<func::FuncOp>(hecate::earth::createProactiveRescaling(
        {waterline, output_val, bootstrap_scale}));
    pm.addNestedPass<func::FuncOp>(hecate::earth::createEarlyModswitch());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addNestedPass<func::FuncOp>(hecate::earth::createLatencyEstimator());
    coverage.addNestedPass<func::FuncOp>(
        hecate::earth::createBootstrapPlacement({bootstrap_scale}));
    coverage.addNestedPass<func::FuncOp>(
        hecate::earth::createCodeSegmentation());
    coverage.addNestedPass<func::FuncOp>(hecate::earth::createCoverageRecorder(
        {waterline, .5f, bootstrap_scale}));
    auto selected = func->getAttrOfType<IntegerAttr>("selected_set").getInt();
    auto initialTargets = ca.getInitialTargets();
    auto segment = [&](int64_t from, int64_t to, const Plan &state,
                       bool initial) {
      auto copy = func.clone();
      copy->setAttr("cutted_edge", builder.getDenseI64ArrayAttr({from, to}));
      auto targets = initial ? initialTargets : ca.getTargets(from, selected);
      copy->setAttr("btp_target", builder.getDenseI64ArrayAttr(targets));
      copy->setAttr("segment_input",
                    builder.getDenseI64ArrayAttr(
                        from ? ca.getValueInfo(from)->getLiveOuts()
                             : SmallVector<int64_t, 4>{}));
      copy->setAttr("segment_inputType", builder.getTypeArrayAttr(state.types));
      copy->setAttr("segment_return", builder.getDenseI64ArrayAttr(
                                          ca.getValueInfo(to)->getLiveOuts()));
      module->push_back(copy);
      return copy;
    };
    for (auto to : ca.getCandidates()) {
      if (to) {
        for (auto from : ca.toFromMap[to]) {
          auto predecessors = plans.find(from);
          if (predecessors == plans.end())
            continue;
          for (const auto &state : predecessors->second)
            for (bool initial : {false, true}) {
              if (initial && (from || initialTargets.empty()))
                continue;
              auto copy = segment(from, to, state, initial);
              LogicalResult feasible = failure();
              {
                ScopedDiagnosticHandler diagnostics(
                    func.getContext(), [](Diagnostic &) { return success(); });
                feasible = pm.run(*module);
              }
              if (succeeded(feasible)) {
                Plan candidate = state;
                candidate.cost += copy->getAttrOfType<FloatAttr>("est_latency")
                                      .getValueAsDouble();
                if (initial)
                  candidate.cuts.push_back(0);
                candidate.cuts.push_back(to);
                candidate.types = inputs;
                candidate.types.append(copy.getResultTypes().begin(),
                                       copy.getResultTypes().end());
                auto &states = plans[to];
                auto same = llvm::find_if(states, [&](const Plan &other) {
                  return other.types == candidate.types;
                });
                if (same == states.end())
                  states.push_back(std::move(candidate));
                else if (candidate.cost < same->cost)
                  *same = std::move(candidate);
              }
              copy.erase();
            }
        }
      }
      auto states = plans.find(to);
      if (states == plans.end() || to == ca.getRetOpid())
        continue;
      for (const auto &state : states->second)
        for (bool initial : {false, true}) {
          if (initial && (to || initialTargets.empty()))
            continue;
          auto copy = segment(to, ca.getRetOpid(), state, initial);
          if (failed(coverage.run(*module))) {
            copy.erase();
            func.emitError("DaCapo coverage analysis failed");
            return signalPassFailure();
          }
          auto ends =
              copy->getAttrOfType<DenseI64ArrayAttr>("coverages").asArrayRef();
          ca.pushFromCoverage(to, SmallVector<int64_t, 2>(ends));
          copy.erase();
        }
    }
    auto result = plans.find(ca.getRetOpid());
    if (result == plans.end() || result->second.empty()) {
      func.emitError("no feasible DaCapo plan for the threshold Q budget");
      return signalPassFailure();
    }
    auto best = std::min_element(
        result->second.begin(), result->second.end(),
        [](const Plan &a, const Plan &b) { return a.cost < b.cost; });
    SmallVector<Attribute> schedule;
    SmallVector<int64_t> targets;
    for (auto cut : best->cuts) {
      if (cut == ca.getRetOpid())
        continue;
      auto refresh = cut ? ca.getTargets(cut, selected) : initialTargets;
      if (refresh.empty())
        continue;
      SmallVector<int64_t> entry{cut};
      entry.append(refresh);
      schedule.push_back(builder.getDenseI64ArrayAttr(entry));
      targets.append(refresh);
    }
    func->setAttr("btp_schedule", builder.getArrayAttr(schedule));
    func->setAttr("btp_target", builder.getDenseI64ArrayAttr(targets));
    func->setAttr("planned_latency", builder.getF64FloatAttr(best->cost));
    llvm::outs() << "Estimated Latency: " << best->cost / 1000000 << " (sec)\n"
                 << "Number of Bootstrapping: " << targets.size() << '\n';
    markAnalysesPreserved<hecate::CandidateAnalysis>();
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
