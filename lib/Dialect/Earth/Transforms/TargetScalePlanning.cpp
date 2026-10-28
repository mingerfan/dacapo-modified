#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/PassManager.h"
#include "nlohmann/json.hpp"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/CommandLine.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <tuple>

using namespace mlir;
using namespace hecate::earth;

namespace {
llvm::cl::opt<bool>
    targetPlanning("native-target-scale-planning",
                   llvm::cl::desc("Experimental backwards native scale "
                                  "planning with greedy Boot insertion"),
                   llvm::cl::init(false));
llvm::cl::opt<std::string> targetBootSchedule(
    "native-target-bootstrap-schedule",
    llvm::cl::desc("Use a JSON Boot schedule without adding automatic Boots"),
    llvm::cl::init(""));
llvm::cl::opt<std::string> targetScaleGoals(
    "native-target-scale-goals",
    llvm::cl::desc("JSON map from original value opid to target log2(scale)"),
    llvm::cl::init(""));
constexpr double tolerance = 1e-8;
struct State {
  int64_t level;
  double scale;
};
struct Plan {
  Value value;
  int64_t work, level;
  double pre, scale;
  SmallVector<std::shared_ptr<Plan>, 2> inputs;
  double encoding = 0;
};
using P = std::shared_ptr<Plan>;

// Shared ciphertexts delimit single-use arithmetic trees. Plan each tree
// backwards, re-encode its local coefficients and preserve the original DAG.
// Refresh an exhausted input frontier when the next tree cannot fit. This
// greedy experiment does not implement a ciphertext noise model.
class Planner {
  func::FuncOp func;
  double floor;
  int64_t bootScale;
  OpBuilder builder;
  llvm::DenseMap<Value, Value> managed;
  llvm::DenseMap<Value, State> states;
  llvm::DenseSet<Value> boundaries;
  llvm::DenseSet<Value> feedsBoot;
  llvm::DenseSet<Value> terminalRoots;
  llvm::DenseMap<Value, Value> linearSources;
  llvm::DenseMap<Value, double> goals;
  std::vector<double> prefix;
  std::map<std::tuple<const void *, int64_t, uint64_t>, P> memo;
  bool emissionFailed = false;

  double removed(int64_t from, int64_t to) const {
    return prefix.at(to) - prefix.at(from);
  }
  bool fits(int64_t level, double scale) const {
    if (level < 0 || level > EarthDialect::nativeLevelAnchor ||
        scale < floor - tolerance)
      return false;
    double capacity = 0;
    for (int64_t i = 0; i <= EarthDialect::nativeLevelAnchor - level; ++i)
      capacity +=
          std::log2(static_cast<double>(EarthDialect::nativeModuli.at(i)));
    return scale + 32 < capacity;
  }
  bool cipher(Value v) const { return getScaleType(v).isCipher(); }
  std::optional<State> natural(Value v) {
    if (auto it = states.find(v); it != states.end())
      return it->second;
    if (boundaries.count(v))
      return std::nullopt;
    auto *op = v.getDefiningOp();
    if (!op)
      return std::nullopt;
    SmallVector<State, 2> args;
    for (Value arg : op->getOperands()) {
      if (!cipher(arg))
        continue;
      auto state = isa<MulOp>(op) ? operandScale(arg) : natural(arg);
      if (!state)
        return std::nullopt;
      args.push_back(*state);
    }
    if (args.empty() || (isa<MulOp>(op) && args.size() != 2))
      return std::nullopt;
    State result = args.front();
    if (args.size() == 2) {
      result.level = std::max(args[0].level, args[1].level);
      if (isa<MulOp>(op))
        result.scale += args[1].scale;
      else if (isa<AddOp>(op)) {
        double gap = std::abs(args[0].scale - args[1].scale);
        if (gap > tolerance && gap < floor - tolerance)
          return std::nullopt;
        result.scale = std::max(args[0].scale, args[1].scale);
      } else
        return std::nullopt;
    }
    return result;
  }
  std::optional<State> operandScale(Value value) {
    auto state = natural(value);
    if (state && !states.count(value)) {
      auto normalized =
          normalizeScaleState({state->level, state->scale}, floor);
      return State{normalized.level, normalized.scale};
    }
    return state;
  }
  int64_t earliest(Value v) {
    if (auto it = states.find(v); it != states.end())
      return it->second.level;
    int64_t result = 0;
    if (auto *op = v.getDefiningOp())
      for (Value arg : op->getOperands())
        if (cipher(arg))
          result = std::max(result, earliest(arg));
    return result;
  }
  void frontier(Value v, llvm::DenseSet<Value> &visited,
                SmallVectorImpl<Value> &inputs) {
    if (!cipher(v) || !visited.insert(v).second)
      return;
    if (states.count(v)) {
      inputs.push_back(v);
      return;
    }
    for (Value arg : v.getDefiningOp()->getOperands())
      frontier(arg, visited, inputs);
  }
  int64_t budget(Value root) const {
    // Fixed schedules and goal-guided terminal suffixes need not retain Q
    // for another Boot. Final outputs can always use their remaining prefix.
    bool output = llvm::any_of(root.getUsers(), [](Operation *user) {
      return isa<func::ReturnOp>(user);
    });
    return !feedsBoot.count(root) &&
                   (!targetBootSchedule.empty() || output ||
                    (!targetScaleGoals.empty() && terminalRoots.count(root)))
               ? EarthDialect::nativeLevelAnchor
               : bootstrapQBudget();
  }
  P planRoot(Value root, std::optional<double> goal) {
    memo.clear();
    if (auto it = goals.find(root); it != goals.end())
      goal = it->second;
    if (!goal)
      if (auto state = natural(root)) {
        auto target =
            isa<RotateOp, NegateOp>(root.getDefiningOp())
                ? ScaleState{state->level, state->scale}
                : normalizeScaleState({state->level, state->scale}, floor);
        if (target.level > budget(root) ||
            (targetBootSchedule.empty() && budget(root) == bootstrapQBudget() &&
             !canBootstrap(target)))
          return {};
        return solve(root, target.level, target.scale);
      }
    for (int64_t level = earliest(root); level <= budget(root); ++level)
      for (unsigned step = 0; step <= (goal ? 0 : 200); ++step)
        // Keep repeated floating-point prime subtraction above the floor.
        // This margin is smaller than the native Boot scale tolerance.
        if (auto plan =
                solve(root, level, goal.value_or(floor + step / 10.) + 1e-9))
          if (!targetBootSchedule.empty() ||
              budget(root) > bootstrapQBudget() ||
              canBootstrap({plan->level, plan->scale}))
            return plan;
    return {};
  }
  LogicalResult refresh(Value logical) {
    // Refresh a shared source once, then replay its linear users. This keeps
    // rotations and x+x from creating separate Boots of the same frontier.
    if (auto source = linearSources.find(logical);
        source != linearSources.end() && states.count(source->second)) {
      Value original = source->second;
      auto state = states.lookup(original);
      if (canBootstrap({state.level, state.scale})) {
        auto fresh = normalizeScaleState(
            {bootstrapOutputLevel(), double(bootScale)}, floor);
        if (state.level > fresh.level ||
            std::abs(state.scale - fresh.scale) > tolerance)
          if (failed(refresh(original)))
            return failure();
        Value input = managed.lookup(original);
        builder.setInsertionPointAfterValue(input);
        auto *copy = builder.clone(*logical.getDefiningOp());
        for (OpOperand &arg : copy->getOpOperands())
          arg.set(input);
        copy->removeAttr("earth.actual_scale");
        SmallVector<Type> types;
        auto inference = cast<InferTypeOpInterface>(copy);
        if (failed(inference.inferReturnTypes(
                copy->getContext(), copy->getLoc(), copy->getOperands(),
                copy->getAttrDictionary(), copy->getPropertiesStorage(),
                copy->getRegions(), types)))
          return copy->emitError("cannot replay linear operation after Boot");
        copy->getResult(0).setType(types.front());
        copy->setAttr("sm_fixed", builder.getUnitAttr());
        managed[logical] = copy->getResult(0);
        states[logical] = {
            static_cast<int64_t>(getScaleType(copy->getResult(0)).getLevel()),
            actualScale(copy->getResult(0))};
        return success();
      }
    }
    Value input = managed.lookup(logical);
    if (!canBootstrap({static_cast<int64_t>(getScaleType(input).getLevel()),
                       actualScale(input)}))
      return func.emitError(
          "automatic Boot input has insufficient Q/encoding precision");
    builder.setInsertionPointAfterValue(input);
    auto boot = builder.create<BootstrapOp>(input.getLoc(), input,
                                            bootstrapOutputLevel(), bootScale);
    auto result = applyThresholdScale(
        cast<ForwardMgmtInterface>(boot.getOperation()), floor, bootScale);
    if (failed(result))
      return boot.emitError("automatic Boot disagrees with the native profile");
    boot->setAttr("sm_fixed", builder.getUnitAttr());
    managed[logical] = *result;
    states[logical] = {static_cast<int64_t>(getScaleType(*result).getLevel()),
                       actualScale(*result)};
    return success();
  }
  P solve(Value v, int64_t level, double scale) {
    if (!fits(level, scale))
      return {};
    uint64_t bits;
    std::memcpy(&bits, &scale, sizeof(bits));
    auto key = std::make_tuple(v.getAsOpaquePointer(), level, bits);
    if (auto it = memo.find(key); it != memo.end())
      return it->second;
    auto &saved = memo[key];
    if (auto it = states.find(v); it != states.end()) {
      auto state = it->second;
      if (state.level > level)
        return {};
      for (int64_t work = level; work >= state.level; --work) {
        double pre = scale + removed(work, level);
        double delta = pre - state.scale;
        if (!fits(work, pre) || delta < -tolerance ||
            (delta > tolerance && delta < floor - tolerance))
          continue;
        return saved = std::make_shared<Plan>(
                   Plan{v, work, level, pre, scale, {}, delta});
      }
      return {};
    }
    auto *op = v.getDefiningOp();
    if (!op || boundaries.count(v))
      return {};
    SmallVector<Value, 2> args;
    for (Value arg : op->getOperands())
      if (cipher(arg))
        args.push_back(arg);
    if (args.empty())
      return {};
    // Share division after an addition tree. For multiplications prefer
    // discarding unused Q before encoding rather than rescaling each term.
    int64_t first = earliest(v);
    for (int64_t step = 0; step <= level - first; ++step) {
      int64_t work = isa<AddOp>(op) ? first + step : level - step;
      double pre = scale + removed(work, level);
      if (!fits(work, pre))
        continue;
      SmallVector<P, 2> inputs;
      double encoding = 0;
      if (isa<AddOp>(op)) {
        for (Value arg : args)
          inputs.push_back(solve(arg, work, pre));
      } else if (isa<NegateOp, RotateOp>(op)) {
        inputs.push_back(solve(args[0], work, pre));
      } else if (isa<MulOp>(op)) {
        if (args.size() == 1) {
          auto source = operandScale(args[0]);
          // Keep nested coefficient products at a high scale until their
          // shared Rescale. Also consider normalizing the inner tree first.
          SmallVector<double, 2> targets;
          if (source)
            targets.push_back(source->scale);
          else {
            targets.push_back(pre - floor);
            targets.push_back(floor + 1e-9);
          }
          for (double inputScale : targets) {
            double plainScale = pre - inputScale;
            if (plainScale < floor - tolerance)
              continue;
            if (auto input = solve(args[0], work, inputScale)) {
              inputs.push_back(input);
              encoding = plainScale;
              break;
            }
          }
          if (inputs.empty())
            continue;
        } else {
          auto a = operandScale(args[0]), b = operandScale(args[1]);
          if (a) {
            inputs.push_back(solve(args[0], work, a->scale));
            inputs.push_back(solve(args[1], work, pre - a->scale));
          } else if (b) {
            inputs.push_back(solve(args[0], work, pre - b->scale));
            inputs.push_back(solve(args[1], work, b->scale));
          } else
            continue;
        }
      } else
        continue;
      if (llvm::all_of(inputs, [](const P &p) { return bool(p); }))
        return saved = std::make_shared<Plan>(
                   Plan{v, work, level, pre, scale, inputs, encoding});
    }
    return {};
  }
  Value switchLevel(Value value, int64_t level) {
    auto current = getScaleType(value).getLevel();
    if (current == level)
      return value;
    auto op =
        builder.create<ModswitchOp>(value.getLoc(), value, level - current);
    op->setAttr("sm_fixed", builder.getUnitAttr());
    return op;
  }
  Value align(Value value, int64_t nominal, double target) {
    double delta = target - actualScale(value);
    auto up = nominal - getScaleType(value).getScale();
    if (std::abs(delta) <= tolerance)
      delta = 0;
    if (!up && !delta)
      return value;
    auto op = builder.create<UpscaleOp>(value.getLoc(), value, up);
    op->setAttr("earth.scale_delta", builder.getF64FloatAttr(delta));
    op->setAttr("sm_fixed", builder.getUnitAttr());
    return op;
  }
  Value encode(Value original, Value cipherValue, double scale) {
    auto constant = original.getDefiningOp<ConstantOp>();
    if (!constant) {
      func.emitError(
          "target scale planning requires locally encoded plaintexts");
      emissionFailed = true;
      return {};
    }
    auto copy = cast<ConstantOp>(builder.clone(*constant));
    copy->getResult(0).setType(
        getScaleType(cipherValue)
            .toPlain()
            .switchScale(static_cast<unsigned>(std::ceil(scale))));
    copy->setAttr("earth.encoding_scale", builder.getF64FloatAttr(scale));
    copy->setAttr("sm_fixed", builder.getUnitAttr());
    return copy;
  }
  Value emit(const P &p) {
    if (auto it = managed.find(p->value); it != managed.end()) {
      Value result = switchLevel(it->second, p->work);
      double delta = p->pre - actualScale(result);
      if (std::abs(delta) <= tolerance)
        delta = 0;
      result = align(result,
                     getScaleType(result).getScale() +
                         static_cast<int64_t>(std::ceil(std::max(0., delta))),
                     p->pre);
      while (getScaleType(result).getLevel() < p->level) {
        auto rescale = builder.create<RescaleOp>(result.getLoc(), result);
        rescale->setAttr("sm_fixed", builder.getUnitAttr());
        result = rescale;
      }
      if (std::abs(actualScale(result) - p->scale) > tolerance) {
        func.emitError(
            "known-value alignment differs from the target scale plan");
        emissionFailed = true;
      }
      return result;
    }
    auto *op = p->value.getDefiningOp();
    SmallVector<Value, 2> original(op->getOperands()), args;
    for (const P &input : p->inputs) {
      builder.setInsertionPoint(op);
      args.push_back(emit(input));
    }
    builder.setInsertionPoint(op);
    if (emissionFailed)
      return {};
    if (isa<AddOp>(op) && args.size() == 2) {
      int64_t nominal = std::max(getScaleType(args[0]).getScale(),
                                 getScaleType(args[1]).getScale());
      for (Value &arg : args)
        arg = align(arg, nominal, p->pre);
    }
    unsigned next = 0;
    for (auto [i, arg] : llvm::enumerate(original)) {
      if (cipher(arg))
        op->setOperand(i, args[next++]);
      else {
        Value plain =
            encode(arg, args.front(), isa<MulOp>(op) ? p->encoding : p->pre);
        if (!plain)
          return {};
        if (isa<AddOp>(op))
          plain.setType(getScaleType(args.front()).toPlain());
        op->setOperand(i, plain);
      }
    }
    if (emissionFailed)
      return {};
    auto inference = cast<InferTypeOpInterface>(op);
    SmallVector<Type> types;
    if (failed(inference.inferReturnTypes(
            op->getContext(), op->getLoc(), op->getOperands(),
            op->getAttrDictionary(), op->getPropertiesStorage(),
            op->getRegions(), types))) {
      op->emitError("target scale type inference failed");
      emissionFailed = true;
      return {};
    }
    op->getResult(0).setType(types.front());
    op->setAttr("sm_fixed", builder.getUnitAttr());
    Value result = op->getResult(0);
    builder.setInsertionPointAfter(op);
    while (getScaleType(result).getLevel() < p->level) {
      auto rescale = builder.create<RescaleOp>(op->getLoc(), result);
      rescale->setAttr("sm_fixed", builder.getUnitAttr());
      result = rescale;
    }
    if (std::abs(actualScale(result) - p->scale) > tolerance) {
      op->emitError("emitted scale differs from the backwards plan");
      emissionFailed = true;
    }
    return result;
  }

public:
  Planner(func::FuncOp func, double floor, int64_t bootScale)
      : func(func), floor(floor), bootScale(bootScale), builder(func) {
    prefix.push_back(0);
    for (int64_t i = EarthDialect::nativeLevelAnchor; i >= 1; --i)
      prefix.push_back(prefix.back() + std::log2(static_cast<double>(
                                           EarthDialect::nativeModuli.at(i))));
    for (Value arg : func.getArguments()) {
      managed[arg] = arg;
      states[arg] = {static_cast<int64_t>(getScaleType(arg).getLevel()),
                     actualScale(arg)};
    }
  }
  LogicalResult run() {
    if (!targetScaleGoals.empty()) {
      std::ifstream input(targetScaleGoals);
      if (!input)
        return func.emitError("cannot open target scale goals");
      auto requested = nlohmann::json::parse(input, nullptr, false);
      if (requested.is_discarded() || !requested.is_object())
        return func.emitError("target scale goals must be a JSON object");
      auto values = attachOpid(func);
      for (auto it = requested.begin(); it != requested.end(); ++it) {
        int64_t id;
        if (llvm::StringRef(it.key()).getAsInteger(10, id) ||
            !it.value().is_number())
          return func.emitError("invalid target scale goal");
        double scale = it.value().get<double>();
        if (id <= 0 || id >= static_cast<int64_t>(values.size()) ||
            !values[id].getDefiningOp() || !cipher(values[id]) ||
            !std::isfinite(scale) || scale < floor)
          return func.emitError("invalid target scale goal");
        goals[values[id]] = scale;
      }
    }
    for (auto &op : llvm::reverse(func.front())) {
      if (isa<BootstrapOp>(op) ||
          llvm::any_of(op.getResults(),
                       [&](Value value) { return feedsBoot.count(value); }))
        for (Value arg : op.getOperands())
          if (cipher(arg))
            feedsBoot.insert(arg);
    }
    SmallVector<Value> roots;
    auto ret = cast<func::ReturnOp>(func.front().getTerminator());
    Operation *lastProduct = nullptr;
    for (auto &op : func.front())
      if (isa<MulOp>(op) &&
          llvm::all_of(op.getOperands(), [&](Value v) { return cipher(v); }))
        lastProduct = &op;
    bool terminal = false;
    for (auto &op : func.front()) {
      if (isa<ConstantOp, func::ReturnOp>(op))
        continue;
      if (!isa<AddOp, MulOp, NegateOp, RotateOp, BootstrapOp>(op))
        return op.emitError("experimental target scale planner supports "
                            "arithmetic and explicit Boot boundaries only");
      Value value = op.getResult(0);
      if (&op == lastProduct)
        terminal = true;
      if (terminal || llvm::is_contained(ret.getOperands(), value))
        terminalRoots.insert(value);
      if (isa<RotateOp, NegateOp>(op) ||
          (isa<AddOp>(op) && op.getOperand(0) == op.getOperand(1)))
        linearSources[value] = op.getOperand(0);
      if (!value.hasOneUse() ||
          llvm::any_of(value.getUsers(),
                       [](Operation *user) {
                         return isa<func::ReturnOp, BootstrapOp>(user);
                       }) ||
          isa<RotateOp, BootstrapOp>(op) || goals.count(value)) {
        boundaries.insert(value);
        roots.push_back(value);
      }
    }
    for (Value root : roots) {
      if (auto boot = root.getDefiningOp<BootstrapOp>()) {
        auto input = managed.lookup(boot.getValue());
        if (!input)
          return boot.emitError("explicit Boot input was not scale-planned");
        boot.setOperand(input);
        auto result = applyThresholdScale(
            cast<ForwardMgmtInterface>(boot.getOperation()), floor, bootScale);
        if (failed(result))
          return boot.emitError(
              "explicit Boot disagrees with the native profile");
        boot->setAttr("sm_fixed", builder.getUnitAttr());
        managed[root] = *result;
        states[root] = {static_cast<int64_t>(getScaleType(*result).getLevel()),
                        actualScale(*result)};
        continue;
      }
      boundaries.erase(root);
      std::optional<double> goal;
      if (llvm::is_contained(ret.getOperands(), root))
        goal = floor;
      if (EarthDialect::nativeBootInputScale > 0 &&
          llvm::any_of(root.getUsers(),
                       [](Operation *user) { return isa<BootstrapOp>(user); }))
        goal = EarthDialect::nativeBootInputScale;
      P plan = planRoot(root, goal);
      if (!plan && targetBootSchedule.empty()) {
        llvm::DenseSet<Value> visited;
        SmallVector<Value> inputs;
        frontier(root, visited, inputs);
        llvm::stable_sort(inputs, [&](Value a, Value b) {
          return states.lookup(a).level > states.lookup(b).level;
        });
        const auto bootState = normalizeScaleState(
            {bootstrapOutputLevel(), double(bootScale)}, floor);
        for (Value input : inputs) {
          const auto state = states.lookup(input);
          if (state.level <= bootState.level &&
              std::abs(state.scale - bootState.scale) <= tolerance)
            continue;
          if (failed(refresh(input)))
            return failure();
          plan = planRoot(root, goal);
          if (plan)
            break;
        }
      }
      boundaries.insert(root);
      if (!plan || plan->level > budget(root))
        return root.getDefiningOp()->emitError(
            "no precision-safe target scale plan within the Boot Q budget, "
            "including after frontier refresh");
      builder.setInsertionPoint(root.getDefiningOp());
      auto value = emit(plan);
      if (emissionFailed)
        return failure();
      managed[root] = value;
      states[root] = {plan->level, plan->scale};
    }
    for (OpOperand &arg : ret->getOpOperands())
      arg.set(managed.lookup(arg.get()));
    return success();
  }
};
} // namespace

bool hecate::earth::usesTargetScalePlanning() { return targetPlanning; }

LogicalResult hecate::earth::prepareTargetBootSchedule(func::FuncOp func,
                                                       int64_t bootScale) {
  if (targetBootSchedule.empty())
    return success();
  std::ifstream input(targetBootSchedule);
  if (!input)
    return func.emitError("cannot open target Boot schedule");
  auto schedule = nlohmann::json::parse(input, nullptr, false);
  if (schedule.is_discarded() || !schedule.is_array())
    return func.emitError("target Boot schedule must be a JSON array");
  OpBuilder builder(func);
  SmallVector<Attribute> entries;
  for (const auto &entry : schedule) {
    if (!entry.is_array() || entry.size() < 2 ||
        !llvm::all_of(
            entry, [](const auto &value) { return value.is_number_integer(); }))
      return func.emitError(
          "target Boot schedule entries require integer cut and target ids");
    auto values = entry.get<std::vector<int64_t>>();
    entries.push_back(builder.getDenseI64ArrayAttr(values));
  }
  func->setAttr("btp_schedule", builder.getArrayAttr(entries));
  PassManager pm(func.getContext(), func::FuncOp::getOperationName());
  pm.addPass(createBootstrapPlacement({bootScale}));
  return pm.run(func);
}

LogicalResult hecate::earth::planTargetScales(func::FuncOp func,
                                              int64_t waterline,
                                              int64_t bootScale) {
  if (!usesNativeScalePlanning())
    return func.emitError(
        "target scale planning requires actual native moduli");
  return Planner(func, waterline, bootScale).run();
}
