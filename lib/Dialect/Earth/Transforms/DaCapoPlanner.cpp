#include "hecate/Dialect/Earth/Analysis/CandidateAnalysis.h"
#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "hecate/Dialect/Earth/Transforms/RegionalScalePlanning.h"
#include "hecate/Support/Support.h"
#include "hecate/Support/CompilationProgress.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "nlohmann/json.hpp"
#include "llvm/Support/CommandLine.h"
#include <fstream>
#include <map>
#include <memory>

namespace hecate::earth {
#define GEN_PASS_DEF_DACAPOPLANNER
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace hecate::earth
using namespace mlir;
namespace {
llvm::cl::opt<bool>
    allRegionalCandidates("regional-all-boot-candidates",
                          llvm::cl::desc("Search all regional Boot candidates "
                                         "instead of increasing frontier sets"),
                          llvm::cl::init(false));
llvm::cl::opt<std::string> searchReport(
    "regional-search-report",
    llvm::cl::desc("Write regional DaCapo candidate states and rejected edges"),
    llvm::cl::init(""));
struct RegionalSegment {
  int64_t from, to;
  SmallVector<int64_t> inputs, outputs;
  OwningOpRef<func::FuncOp> code;
  std::shared_ptr<RegionalSegment> previous;
};

// Retain the exact successful scale plans. Re-running regional search over the
// unsplit function can choose different boundary states from the DP trials.
LogicalResult composeRegionalPlan(func::FuncOp func,
                                  std::shared_ptr<RegionalSegment> last,
                                  TypeRange inputTypes, int64_t waterline) {
  SmallVector<std::shared_ptr<RegionalSegment>> segments;
  for (auto part = last; part; part = part->previous)
    segments.push_back(part);
  std::reverse(segments.begin(), segments.end());
  auto *old = &func.front();
  auto *body = new Block();
  func.getBody().push_back(body);
  OpBuilder builder(func.getContext());
  builder.setInsertionPointToEnd(body);
  DenseMap<int64_t, Value> values;
  for (auto [i, type] : llvm::enumerate(inputTypes))
    values[i + 1] = body->addArgument(type, old->getArgument(i).getLoc());
  SmallVector<Value> finalResults;
  auto trace = nlohmann::json::array();
  for (const auto &part : segments) {
    auto piece = *part->code;
    if (piece.getNumArguments() != inputTypes.size() + part->inputs.size())
      return func.emitError("regional DaCapo segment argument count changed");
    IRMapping mapping;
    for (size_t i = 0; i < inputTypes.size(); ++i)
      mapping.map(piece.getArgument(i), body->getArgument(i));
    for (auto [i, id] : llvm::enumerate(part->inputs)) {
      auto value = values.find(id);
      if (value == values.end())
        return func.emitError("regional DaCapo segment lost a live input");
      mapping.map(piece.getArgument(inputTypes.size() + i), value->second);
    }
    for (auto &op : piece.front().without_terminator())
      builder.clone(op, mapping);
    finalResults.clear();
    for (auto value : piece.front().getTerminator()->getOperands())
      finalResults.push_back(mapping.lookup(value));
    if (!part->outputs.empty()) {
      if (part->outputs.size() != finalResults.size())
        return func.emitError("regional DaCapo segment return count changed");
      for (auto [i, id] : llvm::enumerate(part->outputs))
        values[id] = finalResults[i];
    }
    if (auto rows = piece->getAttrOfType<StringAttr>("earth.regional_trace"))
      for (auto row : nlohmann::json::parse(rows.getValue().str())) {
        row["segment_from"] = part->from;
        row["segment_to"] = part->to;
        trace.push_back(std::move(row));
      }
  }
  builder.create<func::ReturnOp>(func.getLoc(), finalResults);
  old->dropAllDefinedValueUses();
  old->erase();
  func->setAttr("earth.arg_scales",
                builder.getDenseF64ArrayAttr(
                    SmallVector<double>(inputTypes.size(), waterline)));
  func->setAttr("earth.scale_waterline", builder.getF64FloatAttr(waterline));
  func->setAttr("earth.region_scale_policy", builder.getUnitAttr());
  func->setAttr("earth.regional_dacapo", builder.getUnitAttr());
  func->setAttr("earth.regional_trace", builder.getStringAttr(trace.dump(2)));
  hecate::earth::finishThresholdScale(func, builder, inputTypes);
  return hecate::earth::writeRegionalScaleReport(func);
}

struct DaCapoPlannerPass
    : hecate::earth::impl::DaCapoPlannerBase<DaCapoPlannerPass> {
  DaCapoPlannerPass() = default;
  DaCapoPlannerPass(hecate::earth::DaCapoPlannerOptions options) {
    waterline = options.waterline;
    output_val = options.output_val;
    bootstrap_scale = options.bootstrap_scale;
  }
  void runOnOperation() override {
    hecate::CompilationProgress progress("dacapo-search");
    auto func = getOperation();
    if (!searchReport.empty() && !hecate::earth::usesRegionalScalePlanning()) {
      func.emitError(
          "--regional-search-report requires --native-scale-policy=regional");
      return signalPassFailure();
    }
    if (hecate::earth::usesRegionalScalePlanning())
      for (const auto &[name, table] :
           hecate::earth::EarthDialect::getLatencyTable())
        if (table.size() <=
            static_cast<size_t>(
                hecate::earth::EarthDialect::nativeLevelAnchor)) {
          func.emitError("regional DaCapo latency table does not cover the "
                         "native Q chain: ")
              << name;
          return signalPassFailure();
        }
    auto &ca = getAnalysis<hecate::CandidateAnalysis>();
    const bool regional = hecate::earth::usesRegionalScalePlanning();
    auto selected = func->getAttrOfType<IntegerAttr>("selected_set").getInt();
    const int64_t maximum =
        regional ? std::max<size_t>(1, ca.getMaxNumOuts()) : selected;
    const int64_t first =
        regional ? (allRegionalCandidates ? maximum : 1) : selected;
    std::unique_ptr<hecate::earth::RegionalScaleAnalysis> regionalCoverage;
    if (regional)
      regionalCoverage = std::make_unique<hecate::earth::RegionalScaleAnalysis>(
          func, waterline, bootstrap_scale ? bootstrap_scale.getValue()
                                         : hecate::earth::EarthDialect::rescalingFactor);
    for (int64_t size = first; size <= maximum; ++size) {
      if (regional) {
        ca.finalizeCandidates(size);
        func->setAttr(
            "selected_set",
            IntegerAttr::get(IntegerType::get(func.getContext(), 64), size));
        llvm::errs() << "Regional DaCapo candidate set " << size << ": "
                     << ca.getCandidates().size() << " cuts\n";
      }
      auto planned = tryPlan(func, ca, size, regionalCoverage.get());
      if (failed(planned))
        return signalPassFailure();
      if (*planned)
        return;
    }
    func.emitError("no feasible DaCapo plan after single-prefix pruning");
    signalPassFailure();
  }
  FailureOr<bool> tryPlan(func::FuncOp func, hecate::CandidateAnalysis &ca,
                          int64_t selected,
                          hecate::earth::RegionalScaleAnalysis *regionalCoverage) {
    OpBuilder builder(func);
    struct Plan {
      double cost = 0;
      SmallVector<int64_t> cuts;
      SmallVector<SmallVector<int64_t>> refreshes;
      SmallVector<Type, 4> types;
      SmallVector<double, 4> scales;
      std::shared_ptr<RegionalSegment> regional;
    };
    // Keep one cheapest prefix per cut. Its live ciphertext types and actual
    // scales include bypassed values and are replayed in the next segment.
    std::map<int64_t, Plan> plans;
    SmallVector<Type, 4> inputs;
    for (auto argument : func.getArguments())
      inputs.push_back(
          hecate::earth::getScaleType(argument).switchScale(waterline));
    SmallVector<double, 4> inputScales(inputs.size(),
                                       static_cast<double>(waterline));
    plans[0] = {0, {}, {}, inputs, inputScales};
    OwningOpRef<ModuleOp> module = ModuleOp::create(func.getLoc());
    PassManager pm(func.getContext()), coverage(func.getContext());
    if (!hecate::earth::usesRegionalScalePlanning()) {
      pm.addNestedPass<func::FuncOp>(
          hecate::earth::createBootstrapPlacement({bootstrap_scale}));
      pm.addNestedPass<func::FuncOp>(hecate::earth::createCodeSegmentation());
    }
    pm.addNestedPass<func::FuncOp>(hecate::earth::createProactiveRescaling(
        {waterline, output_val, bootstrap_scale}));
    pm.addNestedPass<func::FuncOp>(hecate::earth::createEarlyModswitch());
    pm.addNestedPass<func::FuncOp>(
        hecate::earth::createAbsorbCoefficientScale({waterline}));
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addNestedPass<func::FuncOp>(hecate::earth::createLatencyEstimator());
    if (!hecate::earth::usesRegionalScalePlanning()) {
      coverage.addNestedPass<func::FuncOp>(
          hecate::earth::createBootstrapPlacement({bootstrap_scale}));
      coverage.addNestedPass<func::FuncOp>(
          hecate::earth::createCodeSegmentation());
    }
    coverage.addNestedPass<func::FuncOp>(hecate::earth::createCoverageRecorder(
        {waterline, .5f, bootstrap_scale}));
    auto initialTargets = ca.getInitialTargets();
    // A refresh can change scale even when it cannot increase Q. Keep this
    // alternative: later coefficient-free branches may otherwise misalign.
    const int64_t refreshScale = bootstrap_scale ? bootstrap_scale.getValue()
        : hecate::earth::EarthDialect::rescalingFactor;
    if (hecate::earth::usesRegionalScalePlanning() && refreshScale != waterline)
      for (auto id : ca.getValueInfo(0)->getLiveOuts())
        if (!llvm::is_contained(initialTargets, id))
          initialTargets.push_back(id);
    auto targetSets = [&](int64_t from, bool initial) {
      SmallVector<SmallVector<int64_t, 4>> choices;
      choices.push_back(initial ? initialTargets : ca.getTargets(from, selected));
      if (hecate::earth::usesRegionalScalePlanning() && from && !initial) {
        auto active = ca.getActiveLiveOuts(from);
        if (active.size() <= static_cast<size_t>(selected) && active != choices.front())
          choices.push_back(std::move(active));
      }
      return choices;
    };
    auto segment = [&](int64_t from, int64_t to, const Plan &state,
                       llvm::ArrayRef<int64_t> targets) {
      func::FuncOp copy;
      if (hecate::earth::usesRegionalScalePlanning()) {
        SmallVector<Value> liveInputs, liveOutputs;
        if (from)
          for (auto id : ca.getValueInfo(from)->getLiveOuts())
            liveInputs.push_back(ca.getValueInfo(id)->getValue());
        for (auto id : ca.getValueInfo(to)->getLiveOuts())
          liveOutputs.push_back(ca.getValueInfo(id)->getValue());
        const bool intermediate = to != ca.getRetOpid();
        if (!intermediate)
          liveOutputs.append(
              func.front().getTerminator()->getOperands().begin(),
              func.front().getTerminator()->getOperands().end());
        copy = hecate::earth::cloneRegionalSegment(
            func, liveInputs, liveOutputs, targets, from, bootstrap_scale,
            intermediate);
        if (!copy)
          return copy;
        copy->setAttr("earth.regional_trial", builder.getUnitAttr());
        SmallVector<int64_t> refreshReturns;
        auto target = ca.getTargets(to, selected);
        for (auto [i, id] : llvm::enumerate(ca.getValueInfo(to)->getLiveOuts()))
          if (llvm::is_contained(target, id))
            refreshReturns.push_back(i);
        copy->setAttr("earth.regional_refresh_returns",
                      builder.getDenseI64ArrayAttr(refreshReturns));
      } else {
        copy = func.clone();
      }
      copy->setAttr("cutted_edge", builder.getDenseI64ArrayAttr({from, to}));
      copy->setAttr("btp_target", builder.getDenseI64ArrayAttr(targets));
      copy->setAttr("segment_input",
                    builder.getDenseI64ArrayAttr(
                        from ? ca.getValueInfo(from)->getLiveOuts()
                             : SmallVector<int64_t, 4>{}));
      copy->setAttr("segment_return", builder.getDenseI64ArrayAttr(
                                          ca.getValueInfo(to)->getLiveOuts()));
      copy->setAttr("segment_inputType", builder.getTypeArrayAttr(state.types));
      if (hecate::earth::usesNativeScalePlanning())
        copy->setAttr("segment_input_scales",
                      builder.getDenseF64ArrayAttr(state.scales));
      module->push_back(copy);
      return copy;
    };
    auto observations = nlohmann::json::array();
    auto rejected = nlohmann::json::array();
    unsigned trials = 0;
    hecate::CompilationProgress progress("dacapo-candidate-cuts");
    unsigned completed = 0;
    const auto cuts = ca.getCandidates();
    for (auto to : cuts) {
      progress.update(completed++, cuts.size());
      SmallVector<size_t, 4> refreshedInputs;
      if (to && to != ca.getRetOpid()) {
        auto targets = ca.getTargets(to, selected);
        auto liveOuts = ca.getValueInfo(to)->getLiveOuts();
        for (size_t i = 0; i < liveOuts.size(); ++i)
          if (llvm::is_contained(targets, liveOuts[i]))
            refreshedInputs.push_back(inputs.size() + i);
      }
      if (to) {
        for (auto from : ca.toFromMap[to]) {
          auto predecessor = plans.find(from);
          if (predecessor == plans.end())
            continue;
          const auto &state = predecessor->second;
          for (bool initial : {false, true}) {
            if (initial && (from || initialTargets.empty()))
              continue;
            for (const auto &targets : targetSets(from, initial)) {
              auto copy = segment(from, to, state, targets);
              if (!copy)
                return failure();
              LogicalResult feasible = failure();
              ++trials;
              {
                ScopedDiagnosticHandler diagnostics(
                    func.getContext(), [&](Diagnostic &diagnostic) {
                      if (!searchReport.empty() &&
                          diagnostic.getSeverity() == DiagnosticSeverity::Error) {
                        std::string location;
                        llvm::raw_string_ostream stream(location);
                        diagnostic.getLocation().print(stream);
                        rejected.push_back({{"from", from},
                                            {"to", to},
                                            {"error", diagnostic.str()},
                                            {"location", stream.str()}});
                      }
                      return success();
                    });
                feasible = pm.run(*module);
              }
              if (succeeded(feasible)) {
                Plan candidate = state;
                if (!targets.empty()) {
                  SmallVector<int64_t> entry{from};
                  entry.append(targets.begin(), targets.end());
                  candidate.refreshes.push_back(std::move(entry));
                }
                candidate.cost += copy->getAttrOfType<FloatAttr>("est_latency")
                                      .getValueAsDouble();
                if (initial)
                  candidate.cuts.push_back(0);
                candidate.cuts.push_back(to);
                candidate.types = inputs;
                candidate.types.append(copy.getResultTypes().begin(),
                                       copy.getResultTypes().end());
                candidate.scales = inputScales;
                if (hecate::earth::usesNativeScalePlanning())
                  for (Value value : copy.front().getTerminator()->getOperands())
                    candidate.scales.push_back(hecate::earth::actualScale(value));
                if (hecate::earth::usesNativeScalePlanning() &&
                    !llvm::all_of(refreshedInputs, [&](size_t i) {
                      auto type = cast<hecate::earth::HEScaleTypeInterface>(
                          candidate.types[i]);
                      return hecate::earth::canBootstrap(
                          {static_cast<int64_t>(type.getLevel()),
                           candidate.scales[i]});
                    })) {
                  copy.erase();
                  continue;
                }
                auto current = plans.find(to);
                if (current == plans.end() ||
                    candidate.cost < current->second.cost) {
                  if (hecate::earth::usesRegionalScalePlanning()) {
                    auto compiled = std::make_shared<RegionalSegment>();
                    compiled->from = from;
                    compiled->to = to;
                    if (from)
                      compiled->inputs = ca.getValueInfo(from)->getLiveOuts();
                    compiled->outputs = ca.getValueInfo(to)->getLiveOuts();
                    compiled->code = OwningOpRef<func::FuncOp>(copy.clone());
                    compiled->previous = state.regional;
                    candidate.regional = std::move(compiled);
                  }
                  plans.insert_or_assign(to, std::move(candidate));
                }
              }
              copy.erase();
            }
          }
        }
      }
      auto state = plans.find(to);
      if (!searchReport.empty()) {
        nlohmann::json row = {{"cut", to},
                              {"reachable", state != plans.end()},
                              {"predecessors", ca.toFromMap[to]},
                              {"targets", ca.getTargets(to, selected)}};
        if (state != plans.end()) {
          row["cost"] = state->second.cost;
          row["states"] = nlohmann::json::array();
          for (size_t i = inputs.size(); i < state->second.types.size(); ++i) {
            auto type = cast<hecate::earth::HEScaleTypeInterface>(
                state->second.types[i]);
            row["states"].push_back(
                {{"q", hecate::earth::EarthDialect::nativeLevelAnchor -
                           type.getLevel() + 1},
                 {"s", state->second.scales[i]}});
          }
        }
        observations.push_back(std::move(row));
      }
      if (state == plans.end() || to == ca.getRetOpid())
        continue;
      for (bool initial : {false, true}) {
        if (initial && (to || initialTargets.empty()))
          continue;
        for (const auto &targets : targetSets(to, initial)) {
          if (regionalCoverage) {
            SmallVector<Value> values(func.getArguments().begin(), func.getArguments().end());
            if (to)
              for (auto id : ca.getValueInfo(to)->getLiveOuts())
                values.push_back(ca.getValueInfo(id)->getValue());
            SmallVector<hecate::earth::RegionalScaleState> states;
            bool feasible = true;
            for (auto [i, value] : llvm::enumerate(values)) {
              auto type = cast<hecate::earth::HEScaleTypeInterface>(state->second.types[i]);
              unsigned q = hecate::earth::EarthDialect::nativeLevelAnchor + 1 - type.getLevel();
              double scale = state->second.scales[i];
              if (llvm::is_contained(targets, hecate::getIntegerAttr("opid", value))) {
                if (!hecate::earth::canBootstrap({static_cast<int64_t>(type.getLevel()), scale})) {
                  feasible = false;
                  break;
                }
                q = hecate::earth::EarthDialect::nativeOutputLevel + 1;
                scale = bootstrap_scale;
              }
              states.push_back({value, q, scale});
            }
            if (!feasible)
              continue;
            auto limit = regionalCoverage->coverage(to, states);
            if (failed(limit))
              return failure();
            if (!to && hecate::compilationProgressEnabled())
              llvm::errs() << "[compile] dacapo-initial-coverage limit=" << *limit
                           << " inputs=" << states.size() << " initial_boot=" << initial << '\n';
            ca.pushFromCoverage(to, {*limit, *limit});
            if (!searchReport.empty())
              observations.back()["coverage"] = {*limit, *limit};
            continue;
          }
          auto copy = segment(to, ca.getRetOpid(), state->second, targets);
          if (!copy)
            return failure();
          if (failed(coverage.run(*module))) {
            copy.erase();
            func.emitError("DaCapo coverage analysis failed");
            return failure();
          }
          auto ends =
              copy->getAttrOfType<DenseI64ArrayAttr>("coverages").asArrayRef();
          ca.pushFromCoverage(to, SmallVector<int64_t, 2>(ends));
          if (!searchReport.empty())
            observations.back()["coverage"] = SmallVector<int64_t>(ends);
          copy.erase();
        }
      }
    }
    if (!searchReport.empty()) {
      std::ofstream report(searchReport);
      if (!report) {
        func.emitError("cannot open regional search report");
        return failure();
      }
      report << nlohmann::json({{"candidate_set", selected},
                                {"segment_trials", trials},
                                {"cuts", observations},
                                {"rejections", rejected}})
                    .dump(2)
             << '\n';
    }
    if (hecate::compilationProgressEnabled())
      llvm::errs() << "[compile] dacapo-attempt trials=" << trials
                   << " reachable_cuts=" << plans.size()
                   << " last_reachable=" << plans.rbegin()->first << '\n';
    auto result = plans.find(ca.getRetOpid());
    if (result == plans.end())
      return false;
    const auto &best = result->second;
    SmallVector<Attribute> schedule;
    SmallVector<int64_t> targets;
    if (hecate::earth::usesRegionalScalePlanning()) {
      for (const auto &entry : best.refreshes) {
        schedule.push_back(builder.getDenseI64ArrayAttr(entry));
        targets.append(entry.begin() + 1, entry.end());
      }
    } else {
      for (auto cut : best.cuts) {
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
    }
    func->setAttr("btp_schedule", builder.getArrayAttr(schedule));
    func->setAttr("btp_target", builder.getDenseI64ArrayAttr(targets));
    func->setAttr("planned_latency", builder.getF64FloatAttr(best.cost));
    llvm::outs() << "Estimated Latency: " << best.cost / 1000000 << " (sec)\n"
                 << "Number of Bootstrapping: " << targets.size() << '\n';
    if (hecate::earth::usesRegionalScalePlanning()) {
      if (failed(composeRegionalPlan(func, best.regional, inputs, waterline)))
        return failure();
    } else {
      markAnalysesPreserved<hecate::CandidateAnalysis>();
    }
    return true;
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
