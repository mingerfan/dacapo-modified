#include "hecate/Dialect/Earth/Transforms/RegionalScalePlanning.h"
#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "hecate/Support/BootstrapInputAlignment.h"
#include "hecate/Support/CompilationProgress.h"
#include "hecate/Support/Support.h"
#include "mlir/IR/IRMapping.h"
#include "nlohmann/json.hpp"
#include "llvm/Support/CommandLine.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <tuple>

using namespace mlir;
using namespace hecate::earth;

namespace {
llvm::cl::opt<std::string> regionReport(
    "regional-scale-report",
    llvm::cl::desc("Write the selected regional scale and Boot trace as JSON"),
    llvm::cl::init(""));
llvm::cl::opt<double> regionOutputScale(
    "regional-output-scale",
    llvm::cl::desc(
        "Required final log2(scale); zero leaves final scale unconstrained"),
    llvm::cl::init(0));
llvm::cl::opt<std::string> regionFailureReport(
    "regional-failure-report", llvm::cl::desc("Write a failed greedy region and its input states as JSON"),
    llvm::cl::init(""));
llvm::cl::opt<unsigned> beamWidth(
    "regional-beam-width", llvm::cl::desc("Maximum regional scale states retained at a boundary (1..32)"),
    llvm::cl::init(32));
constexpr unsigned maxNodes = 64, maxDepth = 6;
constexpr double tolerance = 1e-8;
struct State {
  unsigned q;
  double scale;
};
struct Node {
  Value value;
  Operation *op = nullptr;
  SmallVector<unsigned, 2> args;
  ConstantOp plain;
  bool boundary = false;
};
struct Plan;
using PlanPtr = std::shared_ptr<Plan>;
struct Plan {
  unsigned id, q, arithmeticQ, rescales = 0;
  double scale, plain = 0;
  std::array<double, 2> units{0, 0};
  SmallVector<PlanPtr, 2> args;
  unsigned work = 0;
  bool leaf = false;
  std::shared_ptr<std::vector<PlanPtr>> sequence;
};
struct Step {
  int64_t parent;
  PlanPtr plan;
  SmallVector<unsigned, 4> boots;
};
struct Frontier {
  std::map<unsigned, State> values;
  int64_t step = -1;
  unsigned work = 0;
  PlanPtr pending;
};

class Planner {
public:
  Planner(func::FuncOp func, double floor, double bootScale)
      : func(func), floor(floor), bootScale(bootScale),
        chain(EarthDialect::nativeScaleChain),
        anchor(EarthDialect::nativeLevelAnchor),
        minimumQ(EarthDialect::nativeArithmeticLowerLevel + 1) {}

  FailureOr<SmallVector<int64_t>> boundaries() {
    if (failed(build()))
      return failure();
    SmallVector<int64_t> result;
    for (Value arg : func.getArguments())
      result.push_back(hecate::getIntegerAttr("opid", arg));
    for (unsigned root : roots)
      result.push_back(hecate::getIntegerAttr("opid", nodes[root].value));
    return result;
  }

  LogicalResult schedule() {
    if (failed(build()))
      return failure();
    SmallVector<Operation *> order;
    DenseSet<Operation *> emitted;
    SmallVector<bool> visited(nodes.size(), false);
    SmallVector<std::pair<unsigned, bool>> pending;
    for (unsigned root : roots) {
      pending.push_back({root, false});
      while (!pending.empty()) {
        auto [id, ready] = pending.pop_back_val();
        if (visited[id])
          continue;
        if (ready || !nodes[id].op) {
          visited[id] = true;
          if (nodes[id].plain && emitted.insert(nodes[id].plain).second)
            order.push_back(nodes[id].plain);
          if (nodes[id].op && emitted.insert(nodes[id].op).second)
            order.push_back(nodes[id].op);
        } else {
          pending.push_back({id, true});
          for (unsigned input : llvm::reverse(nodes[id].args))
            pending.push_back({input, false});
        }
      }
    }
    // Match the existing regional emitter's dependency order. Otherwise raw
    // interleaved convolution branches create artificial wide live frontiers.
    auto *end = func.front().getTerminator();
    for (auto *op : order)
      op->moveBefore(end);
    return success();
  }

  LogicalResult run(bool insertBoots, int64_t *coverage = nullptr,
                    unsigned coverageMinimumQ = 0, int64_t from = -1,
                    llvm::ArrayRef<RegionalScaleState> inputs = {}) {
    hecate::CompilationProgress progress("regional-greedy", insertBoots);
    if (failed(build()))
      return failure();
    history.clear();
    queries = 0;
    const auto mid = func->getAttrOfType<BoolAttr>("is_mid_segment");
    const double outputScale = regionOutputScale.getValue();
    std::set<unsigned> fixedOutputs;
    if (!mid || !mid.getValue())
      fixedOutputs = returns;
    if (auto outputs = func->getAttrOfType<DenseI64ArrayAttr>("earth.regional_final_returns"))
      for (auto index : outputs.asArrayRef())
        fixedOutputs.insert(ids.lookup(func.front().getTerminator()->getOperand(index)));
    auto productionOutputs = fixedOutputs;
    productionOutputs.insert(outputProducers.begin(), outputProducers.end());
    std::set<unsigned> bootRequired = explicitBootInputs;
    if (auto refreshed = func->getAttrOfType<DenseI64ArrayAttr>(
            "earth.regional_refresh_returns"))
      for (auto index : refreshed.asArrayRef())
        bootRequired.insert(
            ids.lookup(func.front().getTerminator()->getOperand(index)));
    Frontier initial;
    for (Value arg : func.getArguments()) {
      if (arg.use_empty())
        continue;
      auto type = getScaleType(arg);
      initial.values[ids.lookup(arg)] = {
          static_cast<unsigned>(anchor - type.getLevel() + 1),
          actualScale(arg)};
    }
    if (from >= 0) {
      initial.values.clear();
      for (const auto &input : inputs)
        initial.values[ids.lookup(input.value)] = {input.q, input.scale};
    }
    SmallVector<Frontier> beam{initial};
    struct Checkpoint {
      unsigned next;
      Frontier state;
    };
    std::optional<Checkpoint> checkpoint;
    auto fresh = [&](State value) {
      return value.q >= EarthDialect::nativeOutputLevel + 1 &&
             std::abs(value.scale - bootScale) <= tolerance;
    };
    auto refreshCount = [&](const Frontier &state, unsigned next) {
      return llvm::count_if(state.values,
                            [&](const auto &v) {
        return lastArithmeticUse[v.first] >= next && !fresh(v.second);
      });
    };
    auto save = [&](unsigned next) {
      if (next < roots.size() && nodes[roots[next]].op->hasAttr("earth.saved_output"))
        return;
      for (const auto &state : beam) {
        if (!refreshCount(state, next))
          continue;
        if (!llvm::all_of(state.values,
                          [&](const auto &v) {
                            return lastArithmeticUse[v.first] < next || refreshable(v.second);
                          }))
          continue;
        if (!checkpoint ||
            refreshCount(state, next) < refreshCount(checkpoint->state, checkpoint->next) ||
            (refreshCount(state, next) == refreshCount(checkpoint->state, checkpoint->next) &&
             (next > checkpoint->next ||
              (next == checkpoint->next &&
               rank(state) < rank(checkpoint->state)))))
          checkpoint = Checkpoint{next, state};
      }
    };
    unsigned next = 0, bootCount = 0;
    if (from >= 0)
      next = std::upper_bound(roots.begin(), roots.end(), from,
          [&](int64_t cut, unsigned root) {
            return cut < hecate::getIntegerAttr("opid", nodes[root].value);
          }) - roots.begin();
    if (from >= 0) {
      // Coverage cuts advance in program order. Keep only the overlapping
      // lookahead window, reusing exact input-state results across queries.
      if (next == roots.size())
        coveragePlans.clear();
      else
        coveragePlans.erase(coveragePlans.begin(), coveragePlans.lower_bound(roots[next]));
    }
    if (insertBoots)
      save(0);
    while (next < roots.size()) {
      progress.update(next, roots.size());
      SmallVector<Frontier> candidates;
      unsigned root = roots[next];
      // Beam states often differ only in unrelated live values (e.g. saved KV
      // outputs). Solve an identical region input state only once per step.
      LocalPlans temporaryPlans;
      auto &localPlans = (from >= 0 || insertBoots) ? coveragePlans[root] : temporaryPlans;
      for (const auto &state : beam) {
        Solver solver(*this, state.values);
        SmallVector<PlanPtr> choices;
        std::vector<std::pair<unsigned, double>> key;
        for (auto id : regionInputs[root]) {
          auto input = state.values.find(id);
          key.emplace_back(input == state.values.end() ? 0 : input->second.q,
                           input == state.values.end() ? 0 : input->second.scale);
        }
        if (auto found = localPlans.find(key); found != localPlans.end()) {
          choices = found->second;
        } else {
          SmallVector<unsigned> templateNodes;
          std::vector<int64_t> templateKey;
          const bool reusable = !linearGroups.count(root) &&
              !isa<BootstrapOp>(nodes[root].op) && !nodes[root].op->hasAttr("earth.saved_output");
          if (reusable) {
            templateKey = regionKey(root, state.values, templateNodes);
            templateKey.push_back(linearInputs.count(root));
            templateKey.push_back(productionOutputs.count(root) && outputScale > 0);
          }
          auto cached = reusable ? templates.find(templateKey) : templates.end();
          if (cached != templates.end()) {
            choices = remapPlans(cached->second, templateNodes);
          } else {
            if (linearGroups.count(root)) {
              double target = productionOutputs.count(root) && outputScale > 0 ? outputScale
                              : bootRequired.count(root) ? floor
                              : returns.count(root) ? -1 : floor;
              for (unsigned q = solver.available(root); q >= minimumQ; --q)
                if (auto plan = linear(root, state.values, q, target)) {
                  choices.push_back(plan);
                  break;
                }
            } else if (isa<BootstrapOp>(nodes[root].op)) {
              auto input = state.values.find(nodes[root].args.front());
              if (input != state.values.end() && refreshable(input->second)) {
                auto p = std::make_shared<Plan>();
                p->id = root;
                p->q = p->arithmeticQ = EarthDialect::nativeOutputLevel + 1;
                p->scale = bootScale;
                choices.push_back(p);
              }
            } else if (!solver.flexible(root)) {
              if (auto raw = solver.raw(root)) {
                // Linear views do not normalize an already produced value. A new
                // product can share its RS with its linear basis correction.
                if (solver.hasMultiply(root)) {
                  auto normalized = chain.normalize(raw->scale, raw->q - 1, floor);
                  if (raw->q >= normalized.drops + minimumQ) {
                    auto p = std::make_shared<Plan>(*raw);
                    p->rescales += normalized.drops;
                    p->q -= normalized.drops;
                    p->scale = normalized.scale;
                    p->work += 4 * normalized.drops;
                    choices.push_back(p);
                  }
                } else if (raw->q >= minimumQ)
                  choices.push_back(raw);
              }
            } else {
              SmallVector<double> targets{floor};
              if (!linearInputs.count(root)) {
                for (unsigned fraction = 1; fraction < 10; ++fraction)
                  targets.push_back(floor + fraction / 10.0);
                targets.append({floor + 1, floor + 4, floor + 10, floor + 20,
                                2 * floor, 2 * floor + 20, 3 * floor, 4 * floor});
              }
              if (productionOutputs.count(root) && outputScale > 0)
                targets.assign(1, outputScale);
              if (auto required = solver.requiredScale(root))
                if (!productionOutputs.count(root) || outputScale == 0 ||
                    std::abs(*required - outputScale) <= 1e-10)
                  targets.assign(1, *required);
              for (double target : targets)
                for (unsigned q = solver.available(root); q >= minimumQ; --q)
                  if (auto p = solver.solve(root, q, target)) {
                    choices.push_back(p);
                    break;
                  }
            }
            if (reusable) {
              if (templates.size() >= 4096)
                templates.clear();
              templates.emplace(std::move(templateKey), CachedPlans{templateNodes, choices});
            }
          }
          localPlans.emplace(std::move(key), choices);
        }
        for (auto plan : choices) {
          Frontier candidate = state;
          candidate.values[root] = replay(plan, state.values);
          if (bootRequired.count(root) && !refreshable(candidate.values[root]))
            continue;
          for (auto it = candidate.values.begin();
               it != candidate.values.end();)
            if (lastUse[it->first] <= next)
              it = candidate.values.erase(it);
            else
              ++it;
          candidate.work += plan->work;
          candidate.pending = plan;
          candidates.push_back(std::move(candidate));
        }
      }
      if (candidates.empty()) {
        if (!regionFailureReport.empty() &&
            ((insertBoots && !checkpoint) || (coverage && !from && !coverageMinimumQ))) {
          nlohmann::json report = {{"root", root}, {"region", next},
              {"total_regions", roots.size()}, {"minimum_q", minimumQ},
              {"queries", queries}, {"returns", returns.count(root)},
              {"output_scale", outputScale}, {"linear_input", linearInputs.count(root)},
              {"linear_group", linearGroups.count(root)},
              {"boot_output_q", EarthDialect::nativeOutputLevel + 1},
              {"boot_output_s", bootScale}, {"frontiers", nlohmann::json::array()}};
          for (const auto &state : beam) {
            nlohmann::json graph = nlohmann::json::array();
            std::set<unsigned> visited;
            auto visit = [&](auto &&self, unsigned id) -> void {
              if (!visited.insert(id).second)
                return;
              if (auto it = state.values.find(id); it != state.values.end()) {
                graph.push_back({{"id", id}, {"q", it->second.q}, {"s", it->second.scale}});
                return;
              }
              for (auto arg : nodes[id].args)
                self(self, arg);
              graph.push_back({{"id", id}, {"op", nodes[id].op->getName().getStringRef().str()},
                               {"args", nodes[id].args}, {"plain", bool(nodes[id].plain)},
                               {"sm_fixed", nodes[id].op->hasAttr("sm_fixed")},
                               {"boundary", nodes[id].boundary}});
            };
            visit(visit, root);
            report["frontiers"].push_back(graph);
          }
          if (linearGroups.count(root))
            report["linear_members"] = linearGroups[root];
          std::string path = regionFailureReport;
          if (coverage) {
            path += ".initial-coverage.json";
            report["scope"] = "initial unrefreshed coverage boundary, not a compilation failure";
          }
          std::ofstream output(path);
          output << report.dump(2) << '\n';
          if (!output)
            return func.emitError("cannot write regional diagnostic: ") << path;
        }
        if (coverage) {
          *coverage = hecate::getIntegerAttr("opid", nodes[root].value);
          if (!from && !coverageMinimumQ && hecate::compilationProgressEnabled())
            llvm::errs() << "[compile] regional-initial-stop region=" << next
                         << " node=" << root << " linear_group=" << linearGroups.count(root)
                         << " live=" << beam.front().values.size() << '\n';
          return success();
        }
        if (!insertBoots || !checkpoint) {
          auto diagnostic = nodes[root].op->emitError(
              "no feasible regional scale plan in the Q budget");
          diagnostic << "; region=" << next << '/' << roots.size()
                     << ", node=" << root << ", live=" << beam.front().values.size();
          return failure();
        }
        auto selected = std::move(*checkpoint);
        checkpoint.reset();
        Step step{selected.state.step, {}, {}};
        for (auto &[id, value] : selected.state.values) {
          if (lastArithmeticUse[id] < selected.next || fresh(value))
            continue;
          step.boots.push_back(id);
          value = {static_cast<unsigned>(EarthDialect::nativeOutputLevel + 1),
                   bootScale};
        }
        bootCount += step.boots.size();
        if (hecate::compilationProgressEnabled())
          llvm::errs() << "[compile] regional-greedy checkpoint region=" << next
                       << " resume=" << selected.next << " refresh=" << step.boots.size() << '\n';
        history.push_back(std::move(step));
        selected.state.step = history.size() - 1;
        selected.state.work = 0;
        beam.assign(1, std::move(selected.state));
        next = selected.next;
        continue;
      }
      if (coverage && coverageMinimumQ &&
          llvm::all_of(candidates, [&](const Frontier &candidate) {
            return candidate.pending->q < coverageMinimumQ;
          })) {
        *coverage = hecate::getIntegerAttr("opid", nodes[root].value);
        return success();
      }
      llvm::stable_sort(candidates, [&](const Frontier &a, const Frontier &b) {
        return rank(a) < rank(b);
      });
      beam.clear();
      std::set<std::vector<int64_t>> seen;
      for (auto &candidate : candidates) {
        std::vector<int64_t> key;
        for (const auto &[id, value] : candidate.values) {
          key.push_back(id);
          key.push_back(value.q);
          key.push_back(std::llround(value.scale * 1e8));
        }
        if (!seen.insert(std::move(key)).second)
          continue;
        history.push_back({candidate.step, candidate.pending, {}});
        candidate.step = history.size() - 1;
        candidate.pending.reset();
        beam.push_back(std::move(candidate));
        if (beam.size() == beamWidth)
          break;
      }
      ++next;
      if (insertBoots)
        save(next);
      if (insertBoots && checkpoint && checkpoint->next < roots.size())
        coveragePlans.erase(coveragePlans.begin(),
                            coveragePlans.lower_bound(roots[checkpoint->next]));
    }
    if (coverage) {
      *coverage = -1;
      return success();
    }
    if (outputScale > 0)
      for (unsigned id : fixedOutputs)
        if (std::abs(beam.front().values.at(id).scale - outputScale) >
            tolerance)
          return func.emitError(
              "regional plan cannot satisfy the requested final scale");
    if (!regionReport.empty() || func->hasAttr("earth.regional_trial")) {
      nlohmann::json report = nlohmann::json::array();
      SmallVector<int64_t> steps;
      for (int64_t i = beam.front().step; i >= 0; i = history[i].parent)
        steps.push_back(i);
      std::reverse(steps.begin(), steps.end());
      auto states = initial.values;
      for (auto i : steps) {
        const auto &step = history[i];
        if (!step.boots.empty()) {
          nlohmann::json inputs = nlohmann::json::array();
          for (auto id : step.boots) {
            inputs.push_back({{"node", id},
                              {"q", states.at(id).q},
                              {"s", states.at(id).scale}});
            states[id] = {
                static_cast<unsigned>(EarthDialect::nativeOutputLevel + 1),
                bootScale};
          }
          report.push_back({{"boot", inputs}});
        }
        if (auto plan = step.plan) {
          states[plan->id] = {plan->q, plan->scale};
          std::string location;
          llvm::raw_string_ostream stream(location);
          nodes[plan->id].value.getLoc().print(stream);
          nlohmann::json row = {
              {"node", plan->id},
              {"opid", hecate::getIntegerAttr("opid", nodes[plan->id].value)},
              {"q", plan->q},
              {"s", plan->scale},
              {"location", stream.str()}};
          if (plan->sequence) {
            std::map<unsigned, unsigned> depth;
            unsigned maximum = 0;
            for (const auto &op : *plan->sequence) {
              unsigned d = 0;
              for (auto arg : nodes[op->id].args)
                d = std::max(d, depth[arg]);
              if (isa<MulOp>(nodes[op->id].op))
                ++d;
              depth[op->id] = d;
              maximum = std::max(maximum, d);
            }
            row["linear_nodes"] = plan->sequence->size();
            row["cp_depth"] = maximum;
            row["input_q"] = plan->arithmeticQ;
          }
          report.push_back(row);
        }
      }
      func->setAttr("earth.regional_trace",
                    StringAttr::get(func.getContext(), report.dump(2)));
      if (!func->hasAttr("earth.regional_trial") &&
          failed(writeRegionalScaleReport(func)))
        return failure();
    }
    if (failed(emit(beam.front())))
      return failure();
    bootCount = llvm::count_if(
        func.front(), [](Operation &op) { return isa<BootstrapOp>(op); });
    OpBuilder builder(func);
    func->setAttr("earth.regional_boot_count",
                  builder.getI64IntegerAttr(bootCount));
    func->setAttr("earth.regional_regions",
                  builder.getI64IntegerAttr(roots.size()));
    func->setAttr("earth.regional_queries", builder.getI64IntegerAttr(queries));
    if (insertBoots)
      llvm::outs() << "Number of Bootstrapping: " << bootCount << '\n';
    return success();
  }

private:
  bool built = false;
  func::FuncOp func;
  double floor, bootScale;
  const hecate::ModulusChain &chain;
  unsigned anchor, minimumQ;
  SmallVector<Node> nodes;
  llvm::DenseMap<Value, unsigned> ids;
  SmallVector<unsigned> roots, lastUse;
  SmallVector<int64_t> lastArithmeticUse;
  std::set<unsigned> returns;
  std::set<unsigned> explicitBootInputs;
  std::set<unsigned> outputProducers;
  std::set<unsigned> linearInputs;
  std::map<unsigned, SmallVector<unsigned>> linearGroups;
  std::map<unsigned, SmallVector<unsigned>> regionInputs;
  using LocalPlans = std::map<std::vector<std::pair<unsigned, double>>, SmallVector<PlanPtr>>;
  std::map<unsigned, LocalPlans> coveragePlans;
  struct CachedPlans {
    SmallVector<unsigned> nodes;
    SmallVector<PlanPtr> choices;
  };
  std::map<std::vector<int64_t>, CachedPlans> templates;
  std::deque<Step> history;
  uint64_t queries = 0;

  std::vector<int64_t> regionKey(unsigned root, const std::map<unsigned, State> &fixed,
                                  SmallVector<unsigned> &order) const {
    DenseMap<unsigned, unsigned> local;
    std::vector<int64_t> key;
    auto bits = [](double value) {
      int64_t result;
      std::memcpy(&result, &value, sizeof(result));
      return result;
    };
    auto visit = [&](auto &&self, unsigned id) -> unsigned {
      if (auto it = local.find(id); it != local.end())
        return it->second;
      SmallVector<unsigned> args;
      const auto leaf = fixed.find(id);
      if (leaf == fixed.end())
        for (auto arg : nodes[id].args)
          args.push_back(self(self, arg));
      unsigned index = order.size();
      local[id] = index;
      order.push_back(id);
      if (leaf != fixed.end()) {
        key.insert(key.end(), {0, static_cast<int64_t>(leaf->second.q), bits(leaf->second.scale)});
      } else {
        auto *op = nodes[id].op;
        key.push_back(isa<MulOp>(op) ? 1 : isa<AddOp>(op) ? 2 : isa<RotateOp>(op) ? 3 : 4);
        key.push_back(op->hasAttr("sm_fixed"));
        key.push_back(bool(nodes[id].plain));
        if (nodes[id].plain)
          key.push_back(bits(actualScale(nodes[id].plain->getResult(0))));
        key.push_back(args.size());
        key.insert(key.end(), args.begin(), args.end());
      }
      return index;
    };
    visit(visit, root);
    return key;
  }

  SmallVector<PlanPtr> remapPlans(const CachedPlans &cached,
                                 llvm::ArrayRef<unsigned> order) const {
    assert(cached.nodes.size() == order.size());
    DenseMap<unsigned, unsigned> ids;
    for (auto [i, old] : llvm::enumerate(cached.nodes))
      ids[old] = order[i];
    DenseMap<Plan *, PlanPtr> copies;
    auto clone = [&](auto &&self, const PlanPtr &plan) -> PlanPtr {
      if (auto it = copies.find(plan.get()); it != copies.end())
        return it->second;
      assert(!plan->sequence && ids.count(plan->id));
      auto result = std::make_shared<Plan>(*plan);
      result->id = ids.lookup(plan->id);
      result->args.clear();
      for (const auto &arg : plan->args)
        result->args.push_back(self(self, arg));
      copies[plan.get()] = result;
      return result;
    };
    SmallVector<PlanPtr> result;
    for (const auto &plan : cached.choices)
      result.push_back(clone(clone, plan));
    return result;
  }

  bool legal(unsigned q, double scale) const {
    return q >= minimumQ && q <= anchor + 1 && std::isfinite(scale) &&
           scale >= floor - 1e-10 && scale <= 1023 &&
           scale + 1 < chain.capacity(q - 1);
  }
  double removed(unsigned high, unsigned low) const {
    return chain.removedLog2(high - 1, high - low);
  }
  // Carry the floating result of the actual operation sequence, rather than
  // its requested target. Tiny discrepancies are amplified by high-degree
  // polynomials if a target is silently treated as the produced scale.
  State replay(const PlanPtr &plan,
               const std::map<unsigned, State> &fixed) const {
    if (plan->leaf) {
      double scale = fixed.at(plan->id).scale;
      for (unsigned i = 0; i < plan->rescales; ++i)
        scale = chain.rescale(scale, plan->arithmeticQ - i - 1);
      return {plan->q, scale};
    }
    if (isa<BootstrapOp>(nodes[plan->id].op))
      return {plan->q, bootScale};
    double scale;
    if (plan->sequence) {
      auto values = fixed;
      for (auto step : *plan->sequence)
        values[step->id] = replay(step, values);
      scale = values.at(plan->id).scale;
    } else {
      SmallVector<double, 2> args;
      double rawTarget = plan->scale + chain.removedLog2(plan->arithmeticQ - 1,
                                                         plan->rescales);
      for (auto [i, child] : llvm::enumerate(plan->args)) {
        double value = replay(child, fixed).scale;
        if (plan->units[i] > 0)
          value += rawTarget - value;
        args.push_back(value);
      }
      scale = args.front();
      const auto &node = nodes[plan->id];
      if (isa<MulOp>(node.op))
        scale = node.plain ? scale + (rawTarget - scale) : scale + args.back();
      else if (isa<AddOp>(node.op) && args.size() == 2)
        scale = std::max(scale, args.back());
    }
    for (unsigned i = 0; i < plan->rescales; ++i)
      scale = chain.rescale(scale, plan->arithmeticQ - i - 1);
    return {plan->q, scale};
  }
  bool refreshable(State value) const {
    return canBootstrap(
        {static_cast<int64_t>(anchor + 1 - value.q), value.scale});
  }
  std::tuple<int, int, unsigned, double> rank(const Frontier &state) const {
    int least = anchor + 1, total = 0;
    double scale = 0;
    for (const auto &[id, value] : state.values) {
      auto normal = chain.normalize(value.scale, value.q - 1, floor);
      // Match the executable multiplication views, including the arithmetic
      // floor; ranking must not assume RS below the legal Q budget.
      int q = value.q - std::min(normal.drops, value.q - minimumQ);
      least = std::min(least, q);
      total += q;
      scale += value.scale;
    }
    return std::make_tuple(-least, -total, state.work, scale);
  }
  bool acyclicLinear(llvm::ArrayRef<unsigned> members) const {
    DenseMap<unsigned, unsigned> index;
    std::vector<unsigned> parent(members.size());
    std::iota(parent.begin(), parent.end(), 0);
    for (auto [i, id] : llvm::enumerate(members))
      index[id] = i;
    auto find = [&](unsigned id) {
      while (parent[id] != id) {
        parent[id] = parent[parent[id]];
        id = parent[id];
      }
      return id;
    };
    for (auto id : members)
      if (!isa<MulOp>(nodes[id].op))
        for (auto arg : nodes[id].args)
          if (index.count(arg))
            parent[find(index[id])] = find(index[arg]);
    std::vector<SmallVector<unsigned>> outgoing(members.size());
    std::vector<unsigned> degree(members.size(), 0);
    for (auto id : members)
      if (isa<MulOp>(nodes[id].op))
        for (auto arg : nodes[id].args)
          if (index.count(arg)) {
            unsigned a = find(index[arg]), b = find(index[id]);
            outgoing[a].push_back(b);
            ++degree[b];
          }
    SmallVector<unsigned> ready;
    unsigned classes = 0;
    for (unsigned i = 0; i < members.size(); ++i)
      if (find(i) == i) {
        ++classes;
        if (!degree[i])
          ready.push_back(i);
      }
    for (unsigned i = 0; i < ready.size(); ++i)
      for (auto out : outgoing[ready[i]])
        if (!--degree[out])
          ready.push_back(out);
    return ready.size() == classes;
  }
  LogicalResult build() {
    if (beamWidth < 1 || beamWidth > 32)
      return func.emitError("regional-beam-width must be in [1, 32]");
    if (built)
      return success();
    hecate::CompilationProgress progress("regional-build",
        !func->hasAttr("earth.regional_trial"));
    auto add = [&](Value value, Operation *op) {
      ids[value] = nodes.size();
      nodes.push_back({value, op, {}, {}, false});
    };
    for (Value arg : func.getArguments()) {
      if (!getScaleType(arg).isCipher())
        return func.emitError(
            "regional planning requires ciphertext arguments");
      add(arg, nullptr);
      nodes.back().boundary = true;
    }
    for (auto &op : func.front()) {
      if (isa<ConstantOp, func::ReturnOp>(op))
        continue;
      if (!isa<MulOp, AddOp, RotateOp, NegateOp, BootstrapOp>(op))
        return op.emitError("unsupported regional scale operation");
      add(op.getResult(0), &op);
      auto &node = nodes.back();
      if (op.hasAttr("earth.output_scale_source"))
        outputProducers.insert(nodes.size() - 1);
      for (Value input : op.getOperands()) {
        if (getScaleType(input).isCipher())
          node.args.push_back(ids.lookup(input));
        else {
          node.plain = input.getDefiningOp<ConstantOp>();
          if (!node.plain)
            return op.emitError(
                "regional planning requires local plaintext encoding");
        }
      }
      if (isa<BootstrapOp>(op))
        explicitBootInputs.insert(node.args.front());
    }
    for (Value value : func.front().getTerminator()->getOperands()) {
      returns.insert(ids.lookup(value));
      nodes[ids.lookup(value)].boundary = true;
    }
    SmallVector<unsigned> sizes(nodes.size(), 0), depths(nodes.size(), 0);
    for (unsigned id = 0; id < nodes.size(); ++id) {
      auto &n = nodes[id];
      if (!n.op)
        continue;
      unsigned size = 1, depth = 0;
      for (unsigned a : n.args) {
        size += nodes[a].boundary ? 0 : sizes[a];
        depth = std::max(depth, nodes[a].boundary ? 0 : depths[a]);
      }
      if (isa<MulOp>(n.op))
        ++depth;
      n.boundary |= !n.value.hasOneUse() || size >= maxNodes ||
                    depth >= maxDepth || isa<BootstrapOp>(n.op) ||
                    n.op->hasAttr("earth.saved_output") ||
                    n.op->hasAttr("earth.output_scale_source") ||
                    n.op->hasAttr("sm_fixed");
      sizes[id] = size;
      depths[id] = depth;
      if (isa<BootstrapOp>(n.op))
        nodes[n.args.front()].boundary = true;
    }
    // A private, coefficient-free basis still needs its own production state
    // before it multiplies a coefficient-bearing tree. Otherwise a late raw
    // basis product is ModSwitched past primes that could have been removed
    // on its shorter parallel branch.
    auto contains = [&](auto &&self, unsigned id, bool coefficient) -> bool {
      const auto &n = nodes[id];
      if (n.boundary)
        return false;
      if (isa<MulOp>(n.op) && (coefficient ? bool(n.plain) : !n.plain))
        return true;
      return llvm::any_of(
          n.args, [&](unsigned a) { return self(self, a, coefficient); });
    };
    for (auto &n : nodes) {
      if (!n.op || !isa<MulOp>(n.op))
        continue;
      for (unsigned side = 0; side < n.args.size(); ++side) {
        unsigned a = n.args[side];
        bool consumerNeedsBasis =
            n.plain || contains(contains, n.args[1 - side], true);
        if (consumerNeedsBasis && !contains(contains, a, true) &&
            contains(contains, a, false))
          nodes[a].boundary = true;
      }
    }
    // Rotations and reductions form a whole linear region. Keeping each of
    // its fanout branches as an independent beam decision loses the common
    // scale needed by the final sum, even when every branch is feasible.
    std::vector<unsigned> parent(nodes.size());
    std::iota(parent.begin(), parent.end(), 0);
    auto representative = [&](unsigned a) {
      while (parent[a] != a) {
        parent[a] = parent[parent[a]];
        a = parent[a];
      }
      return a;
    };
    auto isLinear = [&](unsigned id) {
      const auto &n = nodes[id];
      return n.op && !isa<BootstrapOp>(n.op) &&
             !n.op->hasAttr("earth.saved_output") &&
             !(isa<MulOp>(n.op) && !n.plain) && !n.op->hasAttr("sm_fixed");
    };
    std::set<unsigned> hard;
    for (unsigned id = 0; id < nodes.size(); ++id) {
      if (returns.count(id))
        hard.insert(id);
      for (Operation *user : nodes[id].value.getUsers())
        if (auto mul = dyn_cast<MulOp>(user);
            (mul && !mul.isSingle()) || user->hasAttr("earth.saved_output"))
          hard.insert(id);
    }
    for (unsigned id = 0; id < nodes.size(); ++id)
      if (isLinear(id))
        for (unsigned arg : nodes[id].args)
          if (isLinear(arg) && !hard.count(arg))
            parent[representative(id)] = representative(arg);
    std::map<unsigned, SmallVector<unsigned>> groups;
    for (unsigned id = 0; id < nodes.size(); ++id)
      if (isLinear(id))
        groups[representative(id)].push_back(id);
    auto registerLinear = [&](const SmallVector<unsigned> &members) {
      std::set<unsigned> memberSet(members.begin(), members.end());
      SmallVector<unsigned> exits;
      bool rotation = false, coefficient = false;
      std::map<unsigned, unsigned> depth;
      unsigned maximumDepth = 0;
      for (unsigned id : members) {
        rotation |= isa<RotateOp>(nodes[id].op);
        coefficient |= isa<MulOp>(nodes[id].op);
        unsigned d = 0;
        for (unsigned a : nodes[id].args)
          d = std::max(d, depth[a]);
        if (isa<MulOp>(nodes[id].op))
          ++d;
        depth[id] = d;
        maximumDepth = std::max(maximumDepth, d);
        bool exit = returns.count(id);
        for (Operation *user : nodes[id].value.getUsers())
          if (user->getNumResults() == 0 || !ids.count(user->getResult(0)) ||
              !memberSet.count(ids.lookup(user->getResult(0))))
            exit = true;
        if (exit)
          exits.push_back(id);
      }
      if ((!rotation && members.size() < maxNodes) || !coefficient ||
          exits.size() != 1 || maximumDepth > maxDepth || !acyclicLinear(members))
        return;
      for (unsigned id : members) {
        nodes[id].boundary = false;
        for (unsigned arg : nodes[id].args)
          if (!memberSet.count(arg)) {
            nodes[arg].boundary = true;
            linearInputs.insert(arg);
          }
      }
      nodes[exits.front()].boundary = true;
      linearGroups[exits.front()] = members;
    };
    for (const auto &[key, members] : groups) {
      const std::set<unsigned> memberSet(members.begin(), members.end());
      SmallVector<unsigned> exits;
      for (unsigned id : members) {
        bool exit = returns.count(id);
        for (Operation *user : nodes[id].value.getUsers())
          if (user->getNumResults() == 0 || !ids.count(user->getResult(0)) ||
              !memberSet.count(ids.lookup(user->getResult(0))))
            exit = true;
        if (exit)
          exits.push_back(id);
      }
      if (exits.size() <= 1) {
        registerLinear(members);
        continue;
      }
      // Sharing a rotation (or a common prefix) must not turn each output's
      // reduction into independently normalized 64-node fragments. Assign
      // private cones to their exits; shared nodes stay materialized once as
      // region inputs. Propagate ownership backwards in topological order,
      // rather than walking each output's full cone separately.
      const unsigned shared = nodes.size();
      DenseMap<unsigned, unsigned> owner;
      for (unsigned id : exits)
        owner[id] = id;
      for (unsigned id : llvm::reverse(members)) {
        auto found = owner.find(id);
        assert(found != owner.end() && "linear node has no reachable exit");
        unsigned target = found->second;
        for (unsigned arg : nodes[id].args)
          if (memberSet.count(arg)) {
            auto [it, inserted] = owner.try_emplace(arg, target);
            if (!inserted && it->second != target)
              it->second = shared;
          }
      }
      std::map<unsigned, SmallVector<unsigned>> privateCones;
      for (unsigned id : members)
        if (owner.lookup(id) != shared)
          privateCones[owner.lookup(id)].push_back(id);
      for (const auto &[exit, cone] : privateCones)
        registerLinear(cone);
    }
    for (unsigned id = 0; id < nodes.size(); ++id)
      if (nodes[id].op && nodes[id].boundary)
        roots.push_back(id);
    // A connected linear component may have several exits or exceed the
    // depth cap (e.g. a Qwen projection plus masked reductions). Its bounded
    // private reduction fragments still need a shared scale/RS solution.
    std::set<unsigned> finalRoots;
    const auto mid = func->getAttrOfType<BoolAttr>("is_mid_segment");
    if (regionOutputScale > 0 && (!mid || !mid.getValue()))
      finalRoots = returns;
    if (regionOutputScale > 0)
      finalRoots.insert(outputProducers.begin(), outputProducers.end());
    if (regionOutputScale > 0)
      if (auto outputs = func->getAttrOfType<DenseI64ArrayAttr>("earth.regional_final_returns"))
        for (auto index : outputs.asArrayRef())
          finalRoots.insert(ids.lookup(func.front().getTerminator()->getOperand(index)));
    for (auto root : roots) {
      if (linearGroups.count(root))
        continue;
      std::set<unsigned> members, inputs;
      bool linearOnly = true, coefficient = false;
      auto visit = [&](auto &&self, unsigned id) -> void {
        if (id != root && nodes[id].boundary) {
          inputs.insert(id);
          return;
        }
        if (!members.insert(id).second)
          return;
        linearOnly &= isLinear(id);
        coefficient |= isa<MulOp>(nodes[id].op) && bool(nodes[id].plain);
        for (auto arg : nodes[id].args)
          self(self, arg);
      };
      visit(visit, root);
      if (linearOnly && coefficient &&
          (members.size() >= maxNodes || finalRoots.count(root))) {
        linearGroups[root].assign(members.begin(), members.end());
        linearInputs.insert(inputs.begin(), inputs.end());
      }
    }
    lastUse.assign(nodes.size(), 0);
    lastArithmeticUse.assign(nodes.size(), -1);
    for (unsigned i = 0; i < roots.size(); ++i) {
      if (auto group = linearGroups.find(roots[i]);
          group != linearGroups.end()) {
        std::set<unsigned> members(group->second.begin(), group->second.end());
        for (unsigned id : group->second)
          for (unsigned arg : nodes[id].args)
            if (!members.count(arg)) {
              lastUse[arg] = i;
              lastArithmeticUse[arg] = i;
              regionInputs[roots[i]].push_back(arg);
            }
        continue;
      }
      auto visit = [&](auto &&self, unsigned id) -> void {
        for (unsigned arg : nodes[id].args)
          if (nodes[arg].boundary) {
            lastUse[arg] = i;
            lastArithmeticUse[arg] = i;
            regionInputs[roots[i]].push_back(arg);
          } else
            self(self, arg);
      };
      visit(visit, roots[i]);
    }
    for (auto &[root, inputs] : regionInputs) {
      llvm::sort(inputs);
      inputs.erase(std::unique(inputs.begin(), inputs.end()), inputs.end());
    }
    for (auto id : returns)
      lastUse[id] = roots.size() + 1;
    built = true;
    return success();
  }

  PlanPtr linear(unsigned root, const std::map<unsigned, State> &fixed,
                 unsigned q, double target) {
    const auto &members = linearGroups.at(root);
    std::map<unsigned, unsigned> index;
    for (unsigned i = 0; i < members.size(); ++i)
      index[members[i]] = i;
    std::vector<unsigned> parent(members.size());
    std::iota(parent.begin(), parent.end(), 0);
    auto representative = [&](unsigned a) {
      while (parent[a] != a) {
        parent[a] = parent[parent[a]];
        a = parent[a];
      }
      return a;
    };
    for (unsigned id : members)
      if (!isa<MulOp>(nodes[id].op))
        for (unsigned arg : nodes[id].args)
          if (index.count(arg))
            parent[representative(index[id])] = representative(index[arg]);
    std::map<unsigned, unsigned> classes;
    std::map<unsigned, unsigned> which;
    for (unsigned id : members) {
      unsigned rep = representative(index[id]);
      if (!classes.count(rep))
        classes[rep] = classes.size();
      which[id] = classes[rep];
    }
    unsigned count = classes.size(), at = anchor + 1;
    std::vector<double> scales(count, floor);
    std::vector<SmallVector<unsigned>> outgoing(count);
    std::vector<SmallVector<double>> externalViews(count);
    std::vector<unsigned> degree(count, 0);
    for (unsigned id : members) {
      unsigned out = which[id];
      bool mul = isa<MulOp>(nodes[id].op);
      for (unsigned arg : nodes[id].args) {
        if (index.count(arg)) {
          if (mul) {
            unsigned in = which[arg];
            if (in == out)
              return {};
            outgoing[in].push_back(out);
            ++degree[out];
          }
        } else {
          auto value = fixed.find(arg);
          if (value == fixed.end())
            return {};
          at = std::min(at, value->second.q);
          scales[out] =
              std::max(scales[out], value->second.scale + (mul ? floor : 0));
          if (!mul)
            externalViews[out].push_back(value->second.scale);
        }
      }
    }
    if (q > at)
      return {};
    SmallVector<unsigned> ready;
    for (unsigned i = 0; i < count; ++i)
      if (!degree[i])
        ready.push_back(i);
    for (unsigned i = 0; i < ready.size(); ++i) {
      unsigned in = ready[i];
      for (double source : externalViews[in])
        if (scales[in] - source > tolerance &&
            scales[in] - source < floor - 1e-10)
          scales[in] = std::max(scales[in], source + floor);
      for (unsigned out : outgoing[in]) {
        scales[out] = std::max(scales[out], scales[in] + floor);
        if (!--degree[out])
          ready.push_back(out);
      }
    }
    if (ready.size() != count)
      return {};
    double product = target < 0 ? scales[which[root]] : target + removed(at, q);
    if (product < scales[which[root]] - 1e-10)
      return {};
    scales[which[root]] = product;
    for (unsigned c = 0; c < count; ++c) {
      if (!legal(at, scales[c]))
        return {};
      for (double source : externalViews[c]) {
        double delta = scales[c] - source;
        if (delta > tolerance && delta < floor - 1e-10)
          return {};
      }
      for (unsigned out : outgoing[c])
        if (scales[out] < scales[c] + floor - 1e-10)
          return {};
    }
    auto result = std::make_shared<Plan>();
    result->id = root;
    result->q = q;
    result->arithmeticQ = at;
    result->scale = target < 0 ? product - removed(at, q) : target;
    result->rescales = at - q;
    result->work = 4 * (at - q);
    result->sequence = std::make_shared<std::vector<PlanPtr>>();
    for (unsigned id : members) {
      auto step = std::make_shared<Plan>();
      step->id = id;
      step->q = step->arithmeticQ = at;
      step->scale = scales[which[id]];
      for (auto [i, arg] : llvm::enumerate(nodes[id].args)) {
        auto leaf = std::make_shared<Plan>();
        leaf->id = arg;
        leaf->q = leaf->arithmeticQ = at;
        leaf->leaf = true;
        leaf->scale =
            index.count(arg) ? scales[which[arg]] : fixed.at(arg).scale;
        step->args.push_back(leaf);
        if (isa<MulOp>(nodes[id].op))
          step->plain = step->scale - leaf->scale;
        else if (step->scale - leaf->scale > tolerance) {
          step->units[i] = step->scale - leaf->scale;
          result->work += 3;
        }
      }
      result->sequence->push_back(step);
    }
    return result;
  }

  class Solver {
  public:
    Solver(Planner &owner, const std::map<unsigned, State> &fixed)
        : p(owner), fixed(fixed) {}
    bool flexible(unsigned id) {
      if (fixed.count(id))
        return false;
      auto it = flex.find(id);
      if (it != flex.end())
        return it->second;
      const auto &n = p.nodes[id];
      bool value = isa<MulOp>(n.op) && n.plain && !n.op->hasAttr("sm_fixed");
      for (auto a : n.args)
        value |= flexible(a);
      return flex[id] = value;
    }
    bool hasMultiply(unsigned id) {
      if (fixed.count(id))
        return false;
      if (auto it = multiplies.find(id); it != multiplies.end())
        return it->second;
      return multiplies[id] = isa<MulOp>(p.nodes[id].op) ||
          llvm::any_of(p.nodes[id].args, [&](unsigned a) { return hasMultiply(a); });
    }
    std::optional<double> requiredScale(unsigned id) {
      if (auto value = fixed.find(id); value != fixed.end())
        return value->second.scale;
      if (auto it = requiredScales.find(id); it != requiredScales.end())
        return it->second;
      if (isa<MulOp>(p.nodes[id].op))
        return requiredScales[id] = std::nullopt;
      for (unsigned arg : p.nodes[id].args)
        if (auto scale = requiredScale(arg))
          return requiredScales[id] = scale;
      return requiredScales[id] = std::nullopt;
    }
    unsigned available(unsigned id) {
      if (auto it = availableQ.find(id); it != availableQ.end())
        return it->second;
      auto it = fixed.find(id);
      if (it != fixed.end())
        return availableQ[id] = it->second.q;
      unsigned q = p.anchor + 1;
      if (auto group = p.linearGroups.find(id); group != p.linearGroups.end()) {
        std::set<unsigned> members(group->second.begin(), group->second.end());
        for (unsigned node : group->second)
          for (unsigned a : p.nodes[node].args)
            if (!members.count(a))
              q = std::min(q, available(a));
        return availableQ[id] = q;
      }
      for (auto a : p.nodes[id].args)
        q = std::min(q, available(a));
      return availableQ[id] = q;
    }
    double naturalScale(unsigned id) {
      if (auto it = naturalScales.find(id); it != naturalScales.end())
        return it->second;
      auto it = fixed.find(id);
      if (it != fixed.end())
        return naturalScales[id] = it->second.scale;
      const auto &n = p.nodes[id];
      double scale = naturalScale(n.args.front());
      if (isa<MulOp>(n.op))
        scale += n.plain ? p.floor : naturalScale(n.args.back());
      else
        for (auto a : n.args)
          scale = std::max(scale, naturalScale(a));
      return naturalScales[id] = scale;
    }
    PlanPtr raw(unsigned id) {
      auto it = rawPlans.find(id);
      if (it != rawPlans.end())
        return it->second;
      auto result = std::make_shared<Plan>();
      result->id = id;
      if (auto v = fixed.find(id); v != fixed.end()) {
        result->q = result->arithmeticQ = v->second.q;
        result->scale = v->second.scale;
        result->leaf = true;
        return rawPlans[id] = result;
      }
      const auto &n = p.nodes[id];
      result->q = result->arithmeticQ = available(id);
      for (auto [i, a] : llvm::enumerate(n.args)) {
        // A square shares its input conversion, including the emitted RS.
        if (i && a == n.args.front()) {
          result->args.push_back(result->args.front());
          continue;
        }
        auto child = raw(a);
        if (!child)
          return rawPlans[id] = {};
        if (isa<MulOp>(n.op)) {
          auto normal = p.chain.normalize(child->scale, child->q - 1, p.floor);
          unsigned drops = std::min(normal.drops, child->q - p.minimumQ);
          if (drops) {
            child = std::make_shared<Plan>(*child);
            child->scale -= p.removed(child->q, child->q - drops);
            child->rescales += drops;
            child->q -= drops;
            child->work += 4 * drops;
          }
        }
        result->q = result->arithmeticQ = std::min(result->q, child->q);
        result->work += child->work;
        result->args.push_back(child);
      }
      result->scale = result->args.front()->scale;
      if (isa<MulOp>(n.op)) {
        if (n.plain) {
          result->plain = std::max(p.floor, actualScale(n.plain->getResult(0)));
          result->scale += result->plain;
        } else
          result->scale += result->args.back()->scale;
      } else if (isa<AddOp>(n.op) && !n.plain) {
        double target = std::max(result->scale, result->args.back()->scale);
        for (unsigned i = 0; i < 2; ++i) {
          double delta = target - result->args[i]->scale;
          if (delta > 1e-10) {
            // These are the correction branches of a coefficient-free basis.
            // An arbitrary sum cannot introduce a precision/RS repair here.
            if (!hasMultiply(id) || delta < p.floor - 1e-10)
              return rawPlans[id] = {};
            result->units[i] = delta;
            result->work += 3;
          }
        }
        result->scale = target;
      }
      if (!p.legal(result->q, result->scale))
        return rawPlans[id] = {};
      return rawPlans[id] = result;
    }
    PlanPtr solve(unsigned id, unsigned q, double scale) {
      ++p.queries;
      if (!p.legal(q, scale) || q > available(id))
        return {};
      // Algebraically identical requests can differ by floating-point ulps
      // after adding/removing native primes. Coalesce only the memo key, at
      // finer precision than the solver's 1e-10 acceptance tolerance. Plans
      // and physical replay retain their actual, unrounded scales.
      auto key = std::make_tuple(id, q, std::llround(scale * 1e12));
      auto it = memo.find(key);
      if (it != memo.end())
        return it->second;
      auto value = search(id, q, scale);
      memo.emplace(key, value);
      return value;
    }

  private:
    Planner &p;
    const std::map<unsigned, State> &fixed;
    std::map<unsigned, bool> flex;
    DenseMap<unsigned, bool> multiplies;
    DenseMap<unsigned, unsigned> availableQ;
    DenseMap<unsigned, double> naturalScales;
    DenseMap<unsigned, std::optional<double>> requiredScales;
    std::map<unsigned, PlanPtr> rawPlans;
    std::map<std::tuple<unsigned, unsigned, int64_t>, PlanPtr> memo;
    std::map<std::tuple<unsigned, unsigned, int64_t>, PlanPtr> operandMemo;

    // Only multiplication may request these views. Addition keeps its existing
    // exact-scale rule and cannot spend Q to lower an already produced input.
    SmallVector<double> operandScales(unsigned id) {
      if (flexible(id))
        return {naturalScale(id), p.floor, p.floor + 10, p.floor + 20, 2 * p.floor};
      SmallVector<double> scales;
      if (auto r = raw(id)) {
        double scale = r->scale;
        unsigned q = r->q;
        scales.push_back(scale);
        while (q > p.minimumQ) {
          scale = p.chain.rescale(scale, q - 1);
          if (!p.legal(--q, scale))
            break;
          scales.push_back(scale);
        }
      }
      return scales;
    }
    PlanPtr operand(unsigned id, unsigned q, double scale) {
      if (flexible(id))
        return solve(id, q, scale);
      if (!p.legal(q, scale))
        return {};
      auto key = std::make_tuple(id, q, std::llround(scale * 1e12));
      if (auto it = operandMemo.find(key); it != operandMemo.end())
        return it->second;
      auto r = raw(id);
      if (!r || q > r->q)
        return operandMemo[key] = {};
      double value = r->scale;
      for (unsigned at = r->q; at >= q; --at) {
        if (std::abs(value - scale) <= 1e-10) {
          auto view = std::make_shared<Plan>(*r);
          view->q = q;
          view->scale = value;
          view->rescales += r->q - at;
          view->work += 4 * (r->q - at);
          return operandMemo[key] = view;
        }
        if (value < scale || at == q)
          break;
        value = p.chain.rescale(value, at - 1);
      }
      return operandMemo[key] = {};
    }
    PlanPtr search(unsigned id, unsigned q, double scale) {
      if (!flexible(id)) {
        auto r = raw(id);
        if (!r || q > r->q || std::abs(r->scale - scale) > 1e-10)
          return {};
        auto result = std::make_shared<Plan>(*r);
        result->q = q;
        return result;
      }
      const auto &n = p.nodes[id];
      auto make = [&](unsigned arithmeticQ, unsigned drops,
                      SmallVector<PlanPtr, 2> children, double plain = 0) {
        auto r = std::make_shared<Plan>();
        r->id = id;
        r->q = q;
        r->arithmeticQ = arithmeticQ;
        r->scale = scale;
        r->rescales = drops;
        r->plain = plain;
        r->args = std::move(children);
        r->work = 4 * drops;
        for (auto [i, a] : llvm::enumerate(r->args))
          if (!i || a != r->args.front())
            r->work += a->work;
        return r;
      };
      if (!isa<MulOp>(n.op)) {
        if (auto required = requiredScale(id))
          if (std::abs(*required - scale) > 1e-10)
            return {};
        SmallVector<PlanPtr, 2> children;
        for (auto a : n.args) {
          auto r = solve(a, q, scale);
          if (!r)
            return {};
          children.push_back(r);
        }
        return make(q, 0, std::move(children));
      }
      PlanPtr best;
      auto consider = [&](PlanPtr candidate) {
        if (candidate && (!best || candidate->work < best->work))
          best = candidate;
      };
      for (unsigned at = q; at <= available(id); ++at) {
        // Every later arithmetic level costs at least this many RS, even
        // with free children. Equal-cost alternatives never replace best.
        if (best && best->work <= 4 * (at - q))
          break;
        double product = scale + p.removed(at, q);
        if (!p.legal(at, product))
          continue;
        if (n.plain) {
          auto inputs = operandScales(n.args.front());
          for (double input : inputs) {
            double plain = product - input;
            if (plain < p.floor - 1e-10)
              continue;
            auto child = operand(n.args.front(), at, input);
            if (child)
              consider(make(at, at - q, {child}, plain));
            if (best && best->work == 4 * (at - q))
              break;
          }
        } else {
          bool fa = flexible(n.args[0]), fb = flexible(n.args[1]);
          SmallVector<double> left;
          if (!fa)
            left = operandScales(n.args[0]);
          else if (!fb)
            for (double right : operandScales(n.args[1]))
              left.push_back(product - right);
          else
            left.append({p.floor, p.floor + 10, p.floor + 20, product / 2});
          for (double a : left) {
            // Both uses of a square share the same converted operand.
            if (n.args[0] == n.args[1] && std::abs(2 * a - product) > 1e-10)
              continue;
            if (!p.legal(at, a) || !p.legal(at, product - a))
              continue;
            auto lhs = operand(n.args[0], at, a);
            if (!lhs)
              continue;
            auto rhs = n.args[0] == n.args[1]
                           ? lhs : operand(n.args[1], at, product - a);
            if (rhs)
              consider(make(at, at - q, {lhs, rhs}));
            if (best && best->work == 4 * (at - q))
              break;
          }
        }
      }
      return best;
    }
  };

  LogicalResult emit(const Frontier &selected) {
    SmallVector<int64_t> steps;
    for (int64_t i = selected.step; i >= 0; i = history[i].parent)
      steps.push_back(i);
    std::reverse(steps.begin(), steps.end());
    auto *old = &func.front();
    auto *block = new Block;
    func.getBody().push_back(block);
    OpBuilder builder(func.getContext());
    builder.setInsertionPointToEnd(block);
    llvm::DenseMap<unsigned, Value> current;
    for (auto [i, arg] : llvm::enumerate(old->getArguments()))
      current[ids.lookup(arg)] =
          block->addArgument(arg.getType(), arg.getLoc());
    func->setAttr("earth.region_scale_policy", builder.getUnitAttr());
    auto drop = [&](Value value, unsigned q) {
      auto actualQ = anchor - getScaleType(value).getLevel() + 1;
      if (actualQ > q)
        value = builder.create<ModswitchOp>(value.getLoc(), value, actualQ - q);
      return value;
    };
    auto encode = [&](ConstantOp original, Value like, double scale, bool one) {
      auto type = getScaleType(like).toPlain().switchScale(std::ceil(scale));
      ConstantOp value;
      if (one) {
        auto payloadType = RankedTensorType::get(
            {EarthDialect::polynomialDegree / 2}, builder.getF64Type());
        auto payload =
            DenseElementsAttr::get(payloadType, builder.getF64FloatAttr(1));
        value =
            builder.create<ConstantOp>(like.getLoc(), type.cast<TensorType>(),
                                       payload, builder.getF64FloatAttr(1));
      } else {
        value = cast<ConstantOp>(builder.clone(*original));
        value.getResult().setType(type.cast<TensorType>());
      }
      value->setAttr("earth.encoding_scale", builder.getF64FloatAttr(scale));
      return value.getResult();
    };
    DenseMap<const Plan *, Value> emittedPlans;
    auto emitPlan = [&](auto &&self, const PlanPtr &plan) -> FailureOr<Value> {
      if (auto it = emittedPlans.find(plan.get()); it != emittedPlans.end())
        return it->second;
      const auto &n = nodes[plan->id];
      if (plan->sequence) {
        Value result;
        for (const auto &step : *plan->sequence) {
          auto value = self(self, step);
          if (failed(value))
            return failure();
          current[step->id] = *value;
        }
        result = current.lookup(plan->id);
        for (unsigned i = 0; i < plan->rescales; ++i)
          result = builder.create<RescaleOp>(n.op->getLoc(), result);
        if (std::abs(actualScale(result) - plan->scale) > tolerance)
          return func.emitError(
              "linear region scale replay disagrees with its plan");
        return result;
      }
      if (plan->leaf) {
        auto it = current.find(plan->id);
        if (it == current.end())
          return failure();
        Value result = drop(it->second, plan->arithmeticQ);
        for (unsigned i = 0; i < plan->rescales; ++i)
          result = builder.create<RescaleOp>(result.getLoc(), result);
        result = drop(result, plan->q);
        if (std::abs(actualScale(result) - plan->scale) > tolerance)
          return func.emitError("multiplication input scale replay disagrees with its plan");
        emittedPlans[plan.get()] = result;
        return result;
      }
      if (n.op->hasAttr("earth.saved_output"))
        return drop(current.lookup(n.args.front()), plan->q);
      if (isa<BootstrapOp>(n.op)) {
        Value input = current.lookup(n.args.front());
        auto boot = builder.create<BootstrapOp>(
            n.op->getLoc(), input, bootstrapOutputLevel(),
            static_cast<int64_t>(bootScale));
        return boot.getResult();
      }
      SmallVector<Value, 2> args;
      for (const auto &[i, child] : llvm::enumerate(plan->args)) {
        auto emitted = self(self, child);
        if (failed(emitted))
          return failure();
        Value value = drop(*emitted, plan->arithmeticQ);
        if (plan->units[i] > 0) {
          double target = plan->scale + chain.removedLog2(plan->arithmeticQ - 1,
                                                          plan->rescales);
          double encoding = target - actualScale(value);
          if (encoding < floor - 1e-9)
            return func.emitError(
                "regional unit encoding is below its precision floor");
          auto plain = encode({}, value, encoding, true);
          value = builder.create<MulOp>(value.getLoc(), value, plain);
        }
        args.push_back(value);
      }
      IRMapping mapping;
      unsigned index = 0;
      for (Value original : n.op->getOperands())
        if (getScaleType(original).isCipher())
          mapping.map(original, args[index++]);
        else {
          double encoding = actualScale(args.front());
          if (isa<MulOp>(n.op)) {
            encoding =
                plan->scale +
                chain.removedLog2(plan->arithmeticQ - 1, plan->rescales) -
                encoding;
            if (encoding < floor - 1e-9)
              return func.emitError(
                  "regional coefficient encoding is below its precision floor");
          }
          Value plain = encode(n.plain, args.front(), encoding, false);
          if (isa<AddOp>(n.op))
            plain.setType(getScaleType(args.front()).toPlain());
          mapping.map(original, plain);
        }
      auto *copy = builder.clone(*n.op, mapping);
      copy->removeAttr("earth.actual_scale");
      unsigned nominal = getScaleType(args.front()).getScale();
      if (isa<MulOp>(copy))
        nominal = getScaleType(copy->getOperand(0)).getScale() +
                  getScaleType(copy->getOperand(1)).getScale();
      if (isa<AddOp>(copy) && args.size() == 2)
        nominal = std::max(nominal, getScaleType(args.back()).getScale());
      copy->getResult(0).setType(
          getScaleType(n.value)
              .switchLevel(anchor + 1 - plan->arithmeticQ)
              .switchScale(nominal));
      Value result = copy->getResult(0);
      for (unsigned i = 0; i < plan->rescales; ++i)
        result = builder.create<RescaleOp>(n.op->getLoc(), result);
      result = drop(result, plan->q);
      if (anchor + 1 - getScaleType(result).getLevel() != plan->q ||
          std::abs(actualScale(result) - plan->scale) > tolerance) {
        copy->emitError(
            "regional scale materialization disagrees with its planned state");
        return failure();
      }
      emittedPlans[plan.get()] = result;
      return result;
    };
    for (int64_t i : steps) {
      const auto &step = history[i];
      for (auto id : step.boots) {
        auto value = current.lookup(id);
        if (!value ||
            !canBootstrap({static_cast<int64_t>(getScaleType(value).getLevel()),
                           actualScale(value)}))
          return func.emitError(
              "selected regional Boot frontier is not refreshable");
        auto boot = builder.create<BootstrapOp>(
            value.getLoc(), value, bootstrapOutputLevel(),
            static_cast<int64_t>(bootScale));
        current[id] = boot.getResult();
      }
      if (step.plan) {
        emittedPlans.clear();
        auto value = emitPlan(emitPlan, step.plan);
        if (failed(value))
          return failure();
        current[step.plan->id] = *value;
      }
    }
    SmallVector<Value> outputs;
    for (Value value : old->getTerminator()->getOperands()) {
      auto replacement = current.lookup(ids.lookup(value));
      if (!replacement)
        return func.emitError("regional plan lost a return value");
      outputs.push_back(replacement);
    }
    builder.create<func::ReturnOp>(func.getLoc(), outputs);
    old->dropAllDefinedValueUses();
    old->erase();
    return success();
  }
};
} // namespace

void hecate::earth::preserveRegionalOutputs(func::FuncOp func) {
  // Give a saved output its own logical lifetime. This zero-offset view is
  // emitted as an alias, so later Boot of the working value cannot replace
  // the saved version and no GPU rotation/copy is introduced.
  auto *ret = func.front().getTerminator();
  OpBuilder builder(func.getContext());
  DenseMap<Value, Value> saved;
  for (auto &operand : ret->getOpOperands()) {
    Value value = operand.get();
    if (auto it = saved.find(value); it != saved.end()) {
      operand.set(it->second);
      continue;
    }
    if (!llvm::any_of(value.getUsers(), [&](Operation *user) { return user != ret; }))
      continue;
    if (auto *producer = value.getDefiningOp()) {
      builder.setInsertionPointAfter(producer);
      producer->setAttr("earth.output_scale_source", builder.getUnitAttr());
    } else
      builder.setInsertionPointToStart(&func.front());
    auto view = builder.create<RotateOp>(value.getLoc(), value, builder.getDenseI64ArrayAttr({0}));
    view->setAttr("earth.saved_output", builder.getUnitAttr());
    saved[value] = view.getResult();
    operand.set(view.getResult());
  }
}

struct hecate::earth::RegionalScaleAnalysis::Impl {
  Planner planner;
  Impl(func::FuncOp func, int64_t waterline, int64_t bootScale)
      : planner(func, waterline, bootScale) {}
};

hecate::earth::RegionalScaleAnalysis::RegionalScaleAnalysis(
    func::FuncOp func, int64_t waterline, int64_t bootScale)
    : impl(std::make_unique<Impl>(func, waterline, bootScale)) {}
hecate::earth::RegionalScaleAnalysis::~RegionalScaleAnalysis() = default;
FailureOr<int64_t> hecate::earth::RegionalScaleAnalysis::coverage(
    int64_t from, llvm::ArrayRef<RegionalScaleState> inputs, unsigned minimumQ) {
  int64_t end = -1;
  if (failed(impl->planner.run(false, &end, minimumQ, from, inputs)))
    return failure();
  return end;
}

LogicalResult hecate::earth::planRegionalScales(func::FuncOp func,
                                                int64_t waterline,
                                                int64_t bootScale,
                                                bool insertBoots) {
  if (!std::isfinite(regionOutputScale) || regionOutputScale < 0 ||
      (regionOutputScale > 0 && regionOutputScale < waterline))
    return func.emitError(
        "regional-output-scale must be zero or at least waterline");
  if (!usesNativeScalePlanning() || !llvm::hasSingleElement(func.getBody()))
    return func.emitError(
        "regional scale planning requires native primes and one block");
  return Planner(func, waterline, bootScale).run(insertBoots);
}

FailureOr<int64_t> hecate::earth::analyzeRegionalScaleCoverage(
    func::FuncOp func, int64_t waterline, int64_t bootScale,
    unsigned minimumQ) {
  int64_t coverage = -1;
  if (failed(
          Planner(func, waterline, bootScale).run(false, &coverage, minimumQ)))
    return failure();
  return coverage;
}

FailureOr<SmallVector<int64_t>>
hecate::earth::regionalScaleBoundaries(func::FuncOp func) {
  return Planner(func, 1, 1).boundaries();
}

LogicalResult hecate::earth::scheduleRegionalRegions(func::FuncOp func) {
  return Planner(func, 1, 1).schedule();
}

func::FuncOp hecate::earth::cloneRegionalSegment(
    func::FuncOp func, ValueRange inputs, ValueRange outputs,
    llvm::ArrayRef<int64_t> bootTargets, int64_t from, int64_t bootScale,
    bool intermediate) {
  auto copy = cast<func::FuncOp>(func->cloneWithoutRegions());
  auto *block = new Block();
  copy.getBody().push_back(block);
  OpBuilder builder(func.getContext());
  builder.setInsertionPointToEnd(block);
  IRMapping mapping;
  DenseMap<int64_t, Value> logical;
  auto argument = [&](Value value) {
    auto arg = block->addArgument(value.getType(), value.getLoc());
    mapping.map(value, arg);
    logical[hecate::getIntegerAttr("opid", value)] = value;
  };
  for (Value value : func.getArguments())
    argument(value);
  for (Value value : inputs)
    argument(value);
  for (auto target : bootTargets) {
    auto value = logical.find(target);
    if (value == logical.end()) {
      func.emitError(
          "regional segment Boot target is not on its input frontier");
      copy.erase();
      return {};
    }
    auto input = mapping.lookup(value->second);
    auto type = getScaleType(input);
    auto boot = builder.create<BootstrapOp>(input.getLoc(), input,
                                            type.getLevel(), type.getScale());
    boot->setAttr("planned_scale", builder.getI64IntegerAttr(bootScale));
    boot->setAttr("boot_target", builder.getI64IntegerAttr(target));
    hecate::setIntegerAttr("opid", boot.getResult(), from);
    mapping.map(value->second, boot.getResult());
  }
  // Discover the needed slice without cloning or canonicalizing the prefix
  // and suffix for every candidate edge.
  DenseSet<Operation *> needed;
  SmallVector<Value> pending(outputs.begin(), outputs.end());
  while (!pending.empty()) {
    auto value = pending.pop_back_val();
    if (mapping.contains(value))
      continue;
    auto *op = value.getDefiningOp();
    if (needed.insert(op).second)
      pending.append(op->getOperands().begin(), op->getOperands().end());
  }
  SmallVector<Operation *> ordered(needed.begin(), needed.end());
  llvm::sort(ordered, [](Operation *a, Operation *b) {
    return a->isBeforeInBlock(b);
  });
  for (auto *op : ordered)
    builder.clone(*op, mapping);
  SmallVector<Value> returns;
  for (Value value : outputs)
    returns.push_back(mapping.lookup(value));
  builder.create<func::ReturnOp>(func.getLoc(), returns);
  copy.setFunctionType(builder.getFunctionType(block->getArgumentTypes(),
                                               ValueRange(returns).getTypes()));
  copy->setAttr("is_mid_segment", builder.getBoolAttr(intermediate));
  SmallVector<int64_t> finalReturns;
  for (auto [i, value] : llvm::enumerate(outputs))
    if (llvm::is_contained(func.front().getTerminator()->getOperands(), value))
      finalReturns.push_back(i);
  copy->setAttr("earth.regional_final_returns", builder.getDenseI64ArrayAttr(finalReturns));
  return copy;
}

LogicalResult hecate::earth::writeRegionalScaleReport(func::FuncOp func) {
  if (regionReport.empty())
    return success();
  auto trace = func->getAttrOfType<StringAttr>("earth.regional_trace");
  if (!trace)
    return func.emitError("missing selected regional scale trace");
  std::ofstream output(regionReport);
  if (!output)
    return func.emitError("cannot open regional scale report");
  output << trace.getValue().str() << '\n';
  return success();
}
