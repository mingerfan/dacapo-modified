#include "hecate/Dialect/Earth/Transforms/RegionalScalePlanning.h"
#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "hecate/Support/BootstrapInputAlignment.h"
#include "mlir/IR/IRMapping.h"
#include "nlohmann/json.hpp"
#include "llvm/Support/CommandLine.h"
#include <algorithm>
#include <array>
#include <cmath>
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
constexpr unsigned maxNodes = 64, maxDepth = 6, beamWidth = 32;
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
        minimumQ(EarthDialect::nativeLowerLevel + 1) {}

  LogicalResult run(bool insertBoots) {
    if (failed(build()))
      return failure();
    Frontier initial;
    for (Value arg : func.getArguments()) {
      if (arg.use_empty())
        continue;
      auto type = getScaleType(arg);
      initial.values[ids.lookup(arg)] = {
          static_cast<unsigned>(anchor - type.getLevel() + 1),
          actualScale(arg)};
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
    auto refreshCount = [&](const Frontier &state) {
      return llvm::count_if(state.values,
                            [&](const auto &v) { return !fresh(v.second); });
    };
    auto save = [&](unsigned next) {
      for (const auto &state : beam) {
        if (!refreshCount(state))
          continue;
        if (!llvm::all_of(state.values,
                          [&](const auto &v) { return refreshable(v.second); }))
          continue;
        if (!checkpoint ||
            refreshCount(state) < refreshCount(checkpoint->state) ||
            (refreshCount(state) == refreshCount(checkpoint->state) &&
             (next > checkpoint->next ||
              (next == checkpoint->next &&
               rank(state) < rank(checkpoint->state)))))
          checkpoint = Checkpoint{next, state};
      }
    };
    unsigned next = 0, bootCount = 0;
    if (insertBoots)
      save(0);
    while (next < roots.size()) {
      SmallVector<Frontier> candidates;
      unsigned root = roots[next];
      for (const auto &state : beam) {
        Solver solver(*this, state.values);
        SmallVector<PlanPtr> choices;
        if (linearGroups.count(root)) {
          double target =
              returns.count(root)
                  ? (regionOutputScale > 0 ? regionOutputScale.getValue() : -1)
                  : floor;
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
          if (returns.count(root) && regionOutputScale > 0)
            targets.assign(1, regionOutputScale);
          if (auto required = solver.requiredScale(root))
            if (!returns.count(root) || regionOutputScale == 0 ||
                std::abs(*required - regionOutputScale) <= 1e-10)
              targets.assign(1, *required);
          for (double target : targets)
            for (unsigned q = solver.available(root); q >= minimumQ; --q)
              if (auto p = solver.solve(root, q, target)) {
                choices.push_back(p);
                break;
              }
        }
        for (auto plan : choices) {
          Frontier candidate = state;
          candidate.values[root] = replay(plan, state.values);
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
        if (!insertBoots || !checkpoint) {
          nodes[root].op->emitError(
              "no feasible regional scale plan in the Q budget");
          return failure();
        }
        auto selected = std::move(*checkpoint);
        checkpoint.reset();
        Step step{selected.state.step, {}, {}};
        for (auto &[id, value] : selected.state.values) {
          if (fresh(value))
            continue;
          step.boots.push_back(id);
          value = {static_cast<unsigned>(EarthDialect::nativeOutputLevel + 1),
                   bootScale};
        }
        bootCount += step.boots.size();
        history.push_back(std::move(step));
        selected.state.step = history.size() - 1;
        selected.state.work = 0;
        beam.assign(1, std::move(selected.state));
        next = selected.next;
        continue;
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
    }
    if (regionOutputScale > 0)
      for (unsigned id : returns)
        if (std::abs(beam.front().values.at(id).scale - regionOutputScale) >
            tolerance)
          return func.emitError(
              "regional plan cannot satisfy the requested final scale");
    if (!regionReport.empty()) {
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
          nlohmann::json row = {{"node", plan->id},
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
      std::ofstream output(regionReport);
      if (!output)
        return func.emitError("cannot open regional scale report");
      output << report.dump(2) << '\n';
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
  func::FuncOp func;
  double floor, bootScale;
  const hecate::ModulusChain &chain;
  unsigned anchor, minimumQ;
  SmallVector<Node> nodes;
  llvm::DenseMap<Value, unsigned> ids;
  SmallVector<unsigned> roots, lastUse;
  std::set<unsigned> returns;
  std::set<unsigned> linearInputs;
  std::map<unsigned, SmallVector<unsigned>> linearGroups;
  std::deque<Step> history;
  uint64_t queries = 0;

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
    if (plan->leaf)
      return {plan->q, fixed.at(plan->id).scale};
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
    return hecate::planBootstrapInputAlignment(
               chain, value.q - 1, value.scale,
               EarthDialect::nativeBootInputLevel,
               EarthDialect::nativeBootInputScale)
        .has_value();
  }
  std::tuple<int, int, unsigned, double> rank(const Frontier &state) const {
    int least = anchor + 1, total = 0;
    double scale = 0;
    for (const auto &[id, value] : state.values) {
      auto normal = chain.normalize(value.scale, value.q - 1, floor);
      int q = value.q - normal.drops;
      least = std::min(least, q);
      total += q;
      scale += value.scale;
    }
    return std::make_tuple(-least, -total, state.work, scale);
  }
  LogicalResult build() {
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
             !(isa<MulOp>(n.op) && !n.plain) && !n.op->hasAttr("sm_fixed");
    };
    std::set<unsigned> hard;
    for (unsigned id = 0; id < nodes.size(); ++id) {
      if (returns.count(id))
        hard.insert(id);
      for (Operation *user : nodes[id].value.getUsers())
        if (auto mul = dyn_cast<MulOp>(user); mul && !mul.isSingle())
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
    for (auto &[key, members] : groups) {
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
          exits.size() != 1 || maximumDepth > maxDepth)
        continue;
      for (unsigned id : members) {
        nodes[id].boundary = false;
        for (unsigned arg : nodes[id].args)
          if (!memberSet.count(arg)) {
            nodes[arg].boundary = true;
            linearInputs.insert(arg);
          }
      }
      nodes[exits.front()].boundary = true;
      linearGroups[exits.front()] = std::move(members);
    }
    for (unsigned id = 0; id < nodes.size(); ++id)
      if (nodes[id].op && nodes[id].boundary)
        roots.push_back(id);
    lastUse.assign(nodes.size(), 0);
    for (unsigned i = 0; i < roots.size(); ++i) {
      if (auto group = linearGroups.find(roots[i]);
          group != linearGroups.end()) {
        std::set<unsigned> members(group->second.begin(), group->second.end());
        for (unsigned id : group->second)
          for (unsigned arg : nodes[id].args)
            if (!members.count(arg))
              lastUse[arg] = i;
        continue;
      }
      auto visit = [&](auto &&self, unsigned id) -> void {
        for (unsigned arg : nodes[id].args)
          if (nodes[arg].boundary)
            lastUse[arg] = i;
          else
            self(self, arg);
      };
      visit(visit, roots[i]);
    }
    for (auto id : returns)
      lastUse[id] = roots.size() + 1;
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
      if (isa<MulOp>(p.nodes[id].op))
        return true;
      return llvm::any_of(p.nodes[id].args,
                          [&](unsigned a) { return hasMultiply(a); });
    }
    std::optional<double> requiredScale(unsigned id) {
      if (auto value = fixed.find(id); value != fixed.end())
        return value->second.scale;
      if (isa<MulOp>(p.nodes[id].op))
        return {};
      for (unsigned arg : p.nodes[id].args)
        if (auto scale = requiredScale(arg))
          return scale;
      return {};
    }
    unsigned available(unsigned id) {
      auto it = fixed.find(id);
      if (it != fixed.end())
        return it->second.q;
      unsigned q = p.anchor + 1;
      if (auto group = p.linearGroups.find(id); group != p.linearGroups.end()) {
        std::set<unsigned> members(group->second.begin(), group->second.end());
        for (unsigned node : group->second)
          for (unsigned a : p.nodes[node].args)
            if (!members.count(a))
              q = std::min(q, available(a));
        return q;
      }
      for (auto a : p.nodes[id].args)
        q = std::min(q, available(a));
      return q;
    }
    double naturalScale(unsigned id) {
      auto it = fixed.find(id);
      if (it != fixed.end())
        return it->second.scale;
      const auto &n = p.nodes[id];
      double scale = naturalScale(n.args.front());
      if (isa<MulOp>(n.op))
        scale += n.plain ? p.floor : naturalScale(n.args.back());
      else
        for (auto a : n.args)
          scale = std::max(scale, naturalScale(a));
      return scale;
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
      for (auto a : n.args) {
        auto child = raw(a);
        if (!child)
          return rawPlans[id] = {};
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
      auto key = std::make_tuple(id, q, scale);
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
    std::map<unsigned, PlanPtr> rawPlans;
    std::map<std::tuple<unsigned, unsigned, double>, PlanPtr> memo;
    PlanPtr search(unsigned id, unsigned q, double scale) {
      if (!flexible(id)) {
        auto r = raw(id);
        if (!r || std::abs(r->scale - scale) > 1e-10)
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
        for (auto a : r->args)
          r->work += a->work;
        return r;
      };
      if (!isa<MulOp>(n.op)) {
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
        double product = scale + p.removed(at, q);
        if (!p.legal(at, product))
          continue;
        if (n.plain) {
          SmallVector<double> inputs{naturalScale(n.args.front()), p.floor,
                                     p.floor + 10, p.floor + 20, 2 * p.floor};
          if (!flexible(n.args.front()))
            inputs.resize(1);
          for (double input : inputs) {
            double plain = product - input;
            if (plain < p.floor - 1e-10)
              continue;
            auto child = solve(n.args.front(), at, input);
            if (child)
              consider(make(at, at - q, {child}, plain));
          }
        } else {
          bool fa = flexible(n.args[0]), fb = flexible(n.args[1]);
          SmallVector<double> left;
          if (!fa)
            left.push_back(naturalScale(n.args[0]));
          else if (!fb)
            left.push_back(product - naturalScale(n.args[1]));
          else
            left.append({p.floor, p.floor + 10, p.floor + 20, product / 2});
          for (double a : left) {
            auto lhs = solve(n.args[0], at, a);
            if (!lhs)
              continue;
            auto rhs = solve(n.args[1], at, product - a);
            if (rhs)
              consider(make(at, at - q, {lhs, rhs}));
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
    auto emitPlan = [&](auto &&self, const PlanPtr &plan) -> FailureOr<Value> {
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
        return drop(it->second, plan->q);
      }
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
