#include "hecate/Support/ScaleAlignmentPlanning.h"
#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <cmath>
#include <map>
#include <memory>
#include <tuple>

namespace hecate::earth {
#define GEN_PASS_DEF_HOISTSCALEALIGNMENT
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
}
using namespace mlir;
using namespace hecate;
using namespace hecate::earth;
namespace {
constexpr double tolerance = 1e-8;
unsigned level(Value v) {
  return EarthDialect::nativeLevelAnchor - getScaleType(v).getLevel();
}
AlignmentState state(Value v) { return {level(v), actualScale(v)}; }
bool unit(ConstantOp c, bool positiveOnly = false) {
  auto data = c ? dyn_cast<DenseElementsAttr>(c.getValue()) : DenseElementsAttr{};
  if (!data || !data.getElementType().isF64() ||
      data.getNumElements() != EarthDialect::polynomialDegree / 2 || !data.isSplat()) return false;
  double value = data.getSplatValue<APFloat>().convertToDouble();
  return value == 1 || (!positiveOnly && value == -1);
}
bool preciseUnit(double scale, double error) {
  if (!std::isfinite(scale) || scale < 0 || scale > 1023) return false;
  double s = std::exp2(scale);
  return std::abs(std::round(s) / s - 1) <= error;
}
std::optional<unsigned> cipherOperand(Operation *op) {
  std::optional<unsigned> result;
  for (auto [i, arg] : llvm::enumerate(op->getOperands()))
    if (getScaleType(arg).isCipher()) {
      if (result) return {};
      result = i;
    }
  return result;
}
bool fixed(Operation *op) {
  return op->hasAttr("sm_fixed") || op->hasAttr("earth.saved_output") ||
         op->hasAttr("earth.output_scale_source");
}
// x*x has two operand uses but only one consumer. It is a private square,
// whereas x used by two distinct operations remains an immutable boundary.
bool privateUse(Value value) {
  if (value.use_empty()) return false;
  Operation *owner = value.use_begin()->getOwner();
  return llvm::all_of(value.getUses(), [&](OpOperand &use) { return use.getOwner() == owner; });
}
struct Query {
  Value input;
  SmallVector<Operation *> origins;
  AlignmentRequest request;
  AlignmentResult result;
};
Query query(Value value, AlignmentState target, double floor, double error,
            const DenseSet<Value> &boundaries) {
  Query q;
  Value cursor = value;
  while (!boundaries.contains(cursor) && privateUse(cursor)) {
    auto *op = cursor.getDefiningOp();
    if (!op || fixed(op) || !isa<MulOp, AddOp, RotateOp, NegateOp, RescaleOp, ModswitchOp, UpscaleOp>(op)) break;
    auto index = cipherOperand(op);
    if (!index) break;
    if (isa<MulOp, AddOp>(op) && !op->getOperand(1 - *index).getDefiningOp<ConstantOp>()) break;
    q.origins.push_back(op);
    cursor = op->getOperand(*index);
  }
  std::reverse(q.origins.begin(), q.origins.end());
  q.input = cursor;
  auto &request = q.request;
  request.input = state(cursor);
  request.target = target;
  request.minimumLevel = EarthDialect::nativeArithmeticLowerLevel;
  request.minimumCiphertextScale = request.minimumPlaintextScale = floor;
  request.maximumUnitError = error;
  for (auto *op : q.origins) {
    AlignmentOperation row;
    row.output = state(op->getResult(0));
    if (isa<MulOp, AddOp>(op)) {
      unsigned index = *cipherOperand(op);
      auto plain = op->getOperand(1 - index).getDefiningOp<ConstantOp>();
      row.kind = isa<MulOp>(op) ? AlignmentKind::MulPlain : AlignmentKind::AddPlain;
      row.plaintextScale = isa<AddOp>(op) ? actualScale(op->getOperand(index)) : actualScale(plain.getResult());
      row.unitPlaintext = unit(plain);
    } else if (auto upscale = dyn_cast<UpscaleOp>(op)) {
      row.kind = AlignmentKind::MulPlain;
      row.plaintextScale = upscale.getUpFactor();
      row.unitPlaintext = true;
    } else if (isa<RotateOp>(op)) row.kind = AlignmentKind::Rotate;
    else if (isa<NegateOp>(op)) row.kind = AlignmentKind::Negate;
    else if (isa<RescaleOp>(op)) row.kind = AlignmentKind::Rescale;
    else row.kind = AlignmentKind::ModSwitch;
    request.operations.push_back(row);
  }
  return q;
}
FailureOr<Value> emitScaleAlignment(
    OpBuilder &builder, Location loc, Value input, const ScaleAlignmentPlan &plan,
    llvm::ArrayRef<Operation *> origins, std::optional<unsigned> exitScale) {
  // ModSwitch, AddCP, Rotate and Negate forward the nominal scale. Give
  // the last scale-producing step the exit type's scale budget.
  const AlignmentStep *scaleProducer = nullptr;
  if (exitScale)
    for (const auto &step : llvm::reverse(plan.steps)) {
      if (step.operation.kind == AlignmentKind::MulPlain ||
          step.operation.kind == AlignmentKind::Rescale) {
        scaleProducer = &step;
        break;
      }
    }
  Value current = input;
  for (const auto &step : plan.steps) {
    const auto &s = step.operation;
    Operation *original = step.origin >= 0 && static_cast<unsigned>(step.origin) < origins.size()
                              ? origins[step.origin] : nullptr;
    if (s.kind == AlignmentKind::MulPlain || s.kind == AlignmentKind::AddPlain) {
      ConstantOp plain;
      if (original && isa<MulOp, AddOp>(original)) {
        auto index = cipherOperand(original);
        if (!index) return failure();
        auto source = original->getOperand(1 - *index).getDefiningOp<ConstantOp>();
        if (!source) return failure();
        plain = cast<ConstantOp>(builder.clone(*source));
      } else {
        if (s.kind != AlignmentKind::MulPlain || !s.unitPlaintext ||
            (step.origin >= 0 && !isa_and_nonnull<UpscaleOp>(original))) return failure();
        auto payloadType = RankedTensorType::get({EarthDialect::polynomialDegree / 2}, builder.getF64Type());
        auto payload = DenseElementsAttr::get(payloadType, builder.getF64FloatAttr(1));
        plain = builder.create<ConstantOp>(loc, getTensorType(current), payload, builder.getF64FloatAttr(1));
      }
      auto type = getScaleType(current).toPlain();
      if (&step == scaleProducer) {
        // Allocate only the nominal bits still needed at the exit.
        // The encoding below still uses the exact, possibly fractional scale.
        if (type.getScale() > *exitScale) return failure();
        type = type.switchScale(*exitScale - type.getScale());
      } else if (s.kind == AlignmentKind::MulPlain) {
        type = type.switchScale(std::ceil(s.plaintextScale));
      }
      plain.getResult().setType(type.cast<TensorType>());
      plain->setAttr("earth.encoding_scale", builder.getF64FloatAttr(s.plaintextScale));
      plain->removeAttr("earth.actual_scale");
      current = s.kind == AlignmentKind::MulPlain
          ? builder.create<MulOp>(loc, current, plain.getResult()).getResult()
          : builder.create<AddOp>(loc, current, plain.getResult()).getResult();
    } else if (s.kind == AlignmentKind::Rescale) {
      if (level(current) != s.output.level + 1) return failure();
      auto rescale = builder.create<RescaleOp>(loc, current);
      current = rescale.getResult();
      if (&step == scaleProducer && getScaleType(current).getScale() != *exitScale) {
        if (*exitScale > getScaleType(rescale.getValue()).getScale()) return failure();
        rescale->setAttr("earth.hoist_exit_scale", builder.getI64IntegerAttr(*exitScale));
        current.setType(getScaleType(current).switchScale(*exitScale));
      }
    } else if (s.kind == AlignmentKind::ModSwitch) {
      if (level(current) <= s.output.level) return failure();
      current = builder.create<ModswitchOp>(loc, current, level(current) - s.output.level);
    } else {
      if (!original || !isa<RotateOp, NegateOp>(original)) return failure();
      IRMapping mapping;
      mapping.map(original->getOperand(0), current);
      auto *copy = builder.clone(*original, mapping);
      copy->removeAttr("earth.actual_scale");
      copy->getResult(0).setType(current.getType());
      current = copy->getResult(0);
    }
    if (level(current) != s.output.level || std::abs(actualScale(current) - s.output.scale) > tolerance)
      return failure();
  }
  return current;
}

// Immutable recipes are selected before touching IR. The numerical chain
// solver remains the authority for CP precision and unit/RS placement.
struct GraphPlan;
using GraphPlanPtr = std::shared_ptr<GraphPlan>;
struct GraphPlan {
  Value source;
  AlignmentState output;
  Query chain;
  Operation *binary = nullptr;
  SmallVector<GraphPlanPtr, 2> inputs;
  unsigned rescaleCost = 0, multiplicationCost = 0, encodingCost = 0;
  std::string strategy = "unchanged";
};
class GraphPlanner {
public:
  GraphPlanner(double floor, double error, const DenseSet<Value> &boundaries,
               unsigned queryLimit, unsigned depthLimit)
      : floor(floor), error(error), boundaries(boundaries),
        queryLimit(queryLimit), depthLimit(depthLimit) {}
  GraphPlanPtr solve(Value value, double target, unsigned depth = 0) {
    if (!legal({level(value), target})) return {};
    if (auto found = memo[value].find(target); found != memo[value].end()) return found->second;
    if (depth >= depthLimit || queries == queryLimit) { limited = true; return {}; }
    ++queries;
    auto reuse = [&](Value v) {
      auto plan = std::make_shared<GraphPlan>();
      plan->source = v;
      plan->output = state(v);
      return plan;
    };
    if (std::abs(actualScale(value)-target) <= tolerance && originalValid(value))
      return memo[value][target] = reuse(value);
    auto q = query(value, {level(value), target}, floor, error, boundaries);
    GraphPlanPtr best;
    auto choose = [&](GraphPlanPtr plan) {
      if (!plan) return;
      auto cost = [](const GraphPlanPtr &p) {
        return std::make_tuple(p->rescaleCost, p->multiplicationCost, p->encodingCost);
      };
      if (!best || cost(plan) < cost(best)) best = std::move(plan);
    };
    // First enumerate local chain candidates without changing its entry value.
    if (originalValid(q.input)) choose(chainPlan(q, reuse(q.input)));
    auto *op = q.input.getDefiningOp();
    bool binary = op && isa<AddOp, MulOp>(op) && op->getNumOperands() == 2 &&
                  getScaleType(op->getOperand(0)).isCipher() && getScaleType(op->getOperand(1)).isCipher();
    if (binary && !boundaries.contains(q.input) && privateUse(q.input) && !fixed(op)) {
      double required = target;
      for (unsigned i = q.request.operations.size(); i > 0; --i) {
        const auto &row = q.request.operations[i-1];
        if (row.kind == AlignmentKind::MulPlain) required -= row.plaintextScale;
        if (row.kind == AlignmentKind::Rescale) {
          unsigned inputLevel = i == 1 ? q.request.input.level : q.request.operations[i-2].output.level;
          required += EarthDialect::nativeScaleChain.removedLog2(inputLevel, inputLevel-row.output.level);
        }
      }
      GraphPlanPtr entry;
      if (q.input != value) entry = solve(q.input, required, depth+1);
      else entry = binaryPlan(op, required, depth+1);
      if (entry) choose(q.origins.empty() ? entry : chainPlan(q, entry));
    }
    // Never return a partially searched plan as if it were the selected result.
    return memo[value][target] = limited ? GraphPlanPtr{} : best;
  }
  bool hitLimit() const { return limited; }
  bool validOriginal(Value value) { return originalValid(value); }
  unsigned queries = 0, candidates = 0;

private:
  double floor, error;
  const DenseSet<Value> &boundaries;
  unsigned queryLimit, depthLimit;
  bool limited = false;
  DenseMap<Value, std::map<double, GraphPlanPtr>> memo;
  DenseMap<Value, bool> valid;
  bool legal(AlignmentState s) const {
    return s.level >= static_cast<unsigned>(EarthDialect::nativeArithmeticLowerLevel) &&
           s.level < EarthDialect::nativeScaleChain.size() && std::isfinite(s.scale) &&
           s.scale >= floor-tolerance && s.scale <= 1023 &&
           s.scale+1 < EarthDialect::nativeScaleChain.capacity(s.level);
  }
  bool originalValid(Value value) {
    // Validation is not a search branch. Use an iterative walk so a long,
    // already valid graph neither consumes the search depth nor the C++ stack.
    SmallVector<std::pair<Value, bool>> pending{{value, false}};
    while (!pending.empty()) {
      auto [current, expanded] = pending.pop_back_val();
      if (valid.count(current)) continue;
      if (!legal(state(current))) { valid[current] = false; continue; }
      auto *op = current.getDefiningOp();
      if (boundaries.contains(current) || !op) { valid[current] = true; continue; }
      if (!expanded) {
        pending.push_back({current, true});
        for (Value arg : op->getOperands())
          if (getScaleType(arg).isCipher() && !valid.count(arg)) pending.push_back({arg, false});
        continue;
      }
      bool accepted = true;
      for (Value arg : op->getOperands()) {
        if (getScaleType(arg).isCipher()) accepted &= valid.lookup(arg);
        else {
          auto c = arg.getDefiningOp<ConstantOp>();
          double scale = actualScale(arg);
          accepted &= c && std::isfinite(scale) && scale+1 < EarthDialect::nativeScaleChain.capacity(level(arg)) &&
              (unit(c) ? preciseUnit(scale, error) : scale >= floor-tolerance);
        }
      }
      if (isa<AddOp>(op))
        accepted &= std::abs(actualScale(op->getOperand(0))-actualScale(op->getOperand(1))) <= tolerance;
      valid[current] = accepted;
    }
    return valid.lookup(value);
  }
  static void addCost(GraphPlan &plan, const GraphPlanPtr &child) {
    plan.rescaleCost += child->rescaleCost;
    plan.multiplicationCost += child->multiplicationCost;
    plan.encodingCost += child->encodingCost;
  }
  GraphPlanPtr chainPlan(Query q, const GraphPlanPtr &input) {
    // Replay the same suffix from its new entry; AddCP must be re-encoded at
    // the ciphertext scale. Original RS positions and physical levels stay put.
    auto current = input->output;
    q.request.input = current;
    for (auto &row : q.request.operations) {
      if (row.kind == AlignmentKind::AddPlain) row.plaintextScale = current.scale;
      if (row.kind == AlignmentKind::MulPlain) current.scale += row.plaintextScale;
      if (row.kind == AlignmentKind::Rescale)
        current.scale = EarthDialect::nativeScaleChain.rescale(current.scale, current.level, current.level-row.output.level);
      current.level = row.output.level;
      row.output = current;
    }
    if (candidates >= 8192) { limited = true; return {}; }
    q.request.maximumCandidates = 8192-candidates;
    q.result = planScaleAlignment(EarthDialect::nativeScaleChain, q.request);
    candidates += q.result.candidates;
    if (q.result.status == AlignmentStatus::SearchLimit) limited = true;
    if (!q.result.plan) return {};
    auto plan = std::make_shared<GraphPlan>();
    plan->source = q.origins.empty() ? q.input : q.origins.back()->getResult(0);
    plan->output = q.result.plan->output;
    plan->inputs.push_back(input);
    plan->rescaleCost = q.result.plan->addedRescaleLevels;
    plan->multiplicationCost = q.result.plan->addedMultiplications;
    // Compare against immutable IR, not the replayed suffix's adjusted biases.
    for (auto &step : q.result.plan->steps)
      if (step.origin >= 0 && (step.operation.kind == AlignmentKind::AddPlain ||
                              step.operation.kind == AlignmentKind::MulPlain)) {
        auto *op = q.origins[step.origin];
        double original = isa<UpscaleOp>(op) ? cast<UpscaleOp>(op).getUpFactor()
            : actualScale(op->getOperand(1-*cipherOperand(op)));
        if (std::abs(original-step.operation.plaintextScale) > tolerance) ++plan->encodingCost;
      }
    addCost(*plan, input);
    plan->strategy = input->strategy == "unchanged" ? q.result.plan->strategy : "graph-hoist";
    plan->chain = std::move(q);
    return plan;
  }
  GraphPlanPtr binaryPlan(Operation *op, double target, unsigned depth) {
    if (!legal({level(op->getResult(0)), target})) return {};
    Value lhs = op->getOperand(0), rhs = op->getOperand(1);
    if (level(lhs) != level(rhs) || level(lhs) != level(op->getResult(0))) return {};
    const double a = actualScale(lhs), b = actualScale(rhs);
    SmallVector<std::pair<double, double>> splits;
    if (isa<AddOp>(op)) splits.push_back({target, target});
    else if (lhs == rhs) splits.push_back({target/2, target/2});
    else {
      double delta = target-a-b;
      splits.append({{a+delta, b}, {a, b+delta}, {a+delta/2, b+delta/2}});
    }
    GraphPlanPtr best;
    for (auto [leftScale, rightScale] : splits) {
      auto left = solve(lhs, leftScale, depth);
      if (!left || limited) continue;
      auto right = lhs == rhs ? left : solve(rhs, rightScale, depth);
      if (!right || limited) continue;
      auto plan = std::make_shared<GraphPlan>();
      plan->source = op->getResult(0);
      plan->binary = op;
      plan->inputs = {left, right};
      double scale = isa<AddOp>(op) ? std::max(left->output.scale, right->output.scale)
                                  : left->output.scale+right->output.scale;
      if (std::abs(scale-target) > tolerance || (isa<AddOp>(op) &&
          std::abs(left->output.scale-right->output.scale) > tolerance)) continue;
      plan->output = {level(plan->source), scale};
      addCost(*plan, left);
      if (right != left) addCost(*plan, right);
      plan->strategy = isa<AddOp>(op) ? "graph-addcc" : lhs == rhs ? "graph-square" : "graph-mulcc";
      auto cost = [](const GraphPlanPtr &p) {
        return std::make_tuple(p->rescaleCost, p->multiplicationCost, p->encodingCost);
      };
      if (!best || cost(plan) < cost(best)) best = plan;
    }
    return best;
  }
};

FailureOr<Value> emitGraphPlan(OpBuilder &builder, Location loc, const GraphPlanPtr &plan,
                              DenseMap<const GraphPlan *, Value> &emitted,
                              std::optional<unsigned> exitScale) {
  if (auto found = emitted.find(plan.get()); found != emitted.end()) return found->second;
  if (plan->inputs.empty()) return plan->source;
  SmallVector<Value, 2> inputs;
  // An aligned AddCC joins equal actual scales. Keep the same nominal budget
  // in the rewritten arms so their max does not inflate the exit type.
  auto childScale = isa_and_nonnull<AddOp>(plan->binary) ? exitScale : std::nullopt;
  if (!plan->binary && llvm::none_of(plan->chain.result.plan->steps, [](const auto &step) {
        return step.operation.kind == AlignmentKind::MulPlain ||
               step.operation.kind == AlignmentKind::Rescale;
      }))
    childScale = exitScale;
  for (const auto &child : plan->inputs) {
    auto input = emitGraphPlan(builder, loc, child, emitted, childScale);
    if (failed(input)) return failure();
    inputs.push_back(*input);
  }
  Value result;
  if (plan->binary) {
    IRMapping mapping;
    for (auto [old, value] : llvm::zip(plan->binary->getOperands(), inputs)) mapping.map(old, value);
    auto *copy = builder.clone(*plan->binary, mapping);
    copy->removeAttr("earth.actual_scale");
    auto inference = cast<InferTypeOpInterface>(copy);
    SmallVector<Type> types;
    if (failed(inference.inferReturnTypes(copy->getContext(), loc, copy->getOperands(),
        copy->getAttrDictionary(), copy->getPropertiesStorage(), copy->getRegions(), types))) return failure();
    result = copy->getResult(0);
    result.setType(types.front());
  } else {
    auto value = emitScaleAlignment(builder, loc, inputs.front(), *plan->chain.result.plan,
                                    plan->chain.origins, exitScale);
    if (failed(value)) return failure();
    result = *value;
  }
  if (level(result) != plan->output.level || std::abs(actualScale(result)-plan->output.scale) > tolerance)
    return failure();
  emitted[plan.get()] = result;
  return result;
}

struct HoistScaleAlignmentPass : hecate::earth::impl::HoistScaleAlignmentBase<HoistScaleAlignmentPass> {
  using Base::Base;
  LogicalResult initialize(MLIRContext *) override {
    // Configure once before nested function passes run in parallel.
    if (!operatorSpecPath.empty()) EarthDialect::setNativeScaleParameters(operatorSpecPath, bootProfile);
    return success();
  }
  LogicalResult repair(func::FuncOp func) {
    clearActualScales(func);
    if (auto scales = func->getAttrOfType<DenseF64ArrayAttr>("earth.arg_scales");
        scales && scales.size() != func.getNumArguments())
      return func.emitError("hoist-scale-alignment: argument scale count mismatch");
    auto validLevel = [&](Value value) {
      auto consumed = getScaleType(value).getLevel();
      return consumed <= EarthDialect::nativeLevelAnchor &&
             level(value) < EarthDialect::nativeScaleChain.size() &&
             level(value) >= static_cast<unsigned>(EarthDialect::nativeArithmeticLowerLevel);
    };
    for (Value v : func.getArguments()) {
      if (!validLevel(v) || !std::isfinite(actualScale(v)) || actualScale(v) < waterline-tolerance ||
          actualScale(v) + 1 >= EarthDialect::nativeScaleChain.capacity(level(v)))
        return func.emitError("hoist-scale-alignment: invalid input scale/level capacity");
    }
    for (auto &op : func.front())
      for (Value v : op.getResults())
        if (!validLevel(v)) return op.emitError("hoist-scale-alignment: invalid physical level");
    DenseSet<Value> boundaries;
    SmallVector<std::pair<Value, AlignmentState>> saved;
    auto save = [&](Value v) {
      if (boundaries.insert(v).second) saved.emplace_back(v, state(v));
    };
    for (Value v : func.getArguments()) save(v);
    for (Value v : func.front().getTerminator()->getOperands()) save(v);
    for (auto &op : func.front()) {
      if (auto boot = dyn_cast<BootstrapOp>(op)) { save(boot.getValue()); save(boot.getResult()); }
      for (Value v : op.getResults())
        if (getScaleType(v).isCipher() && ((!v.use_empty() && !privateUse(v)) || fixed(&op))) save(v);
    }
    OpBuilder builder(func.getContext());
    func->setAttr("earth.scale_alignment_policy", builder.getUnitAttr());
    SmallVector<Attribute> trace;
    if (auto previous = func->getAttrOfType<ArrayAttr>("earth.scale_alignment_trace"))
      trace.append(previous.begin(), previous.end());
    auto apply = [&](Value source, AlignmentState target, Operation *at,
                     std::optional<unsigned> exitScale = std::nullopt) -> FailureOr<Value> {
      GraphPlanner planner(waterline, unitError, boundaries, maxGraphQueries, maxGraphDepth);
      auto plan = planner.solve(source, target.scale);
      if (!plan || planner.hitLimit()) {
        at->emitError("hoist-scale-alignment: ") << (planner.hitLimit()
            ? "graph search limit exceeded; no partial plan committed"
            : "no precision-safe candidate in the fixed-level graph search; no Boot or exit-level relaxation was attempted");
        return failure();
      }
      builder.setInsertionPoint(at);
      DenseMap<const GraphPlan *, Value> emitted;
      auto value = emitGraphPlan(builder, at->getLoc(), plan, emitted, exitScale);
      if (failed(value) || level(*value) != target.level ||
          std::abs(actualScale(*value) - target.scale) > tolerance) {
        at->emitError("hoist-scale-alignment: cannot emit the plan with its required exit state");
        return failure();
      }
      trace.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("strategy", builder.getStringAttr(plan->strategy)),
          builder.getNamedAttr("candidates", builder.getI64IntegerAttr(planner.candidates)),
          builder.getNamedAttr("graph_queries", builder.getI64IntegerAttr(planner.queries)),
          builder.getNamedAttr("level", builder.getI64IntegerAttr(target.level)),
          builder.getNamedAttr("scale", builder.getF64FloatAttr(target.scale)),
          builder.getNamedAttr("added_rescale_levels", builder.getI64IntegerAttr(plan->rescaleCost)),
          builder.getNamedAttr("added_multiplications", builder.getI64IntegerAttr(plan->multiplicationCost)),
          builder.getNamedAttr("reencoded_plaintexts", builder.getI64IntegerAttr(plan->encodingCost))}));
      return *value;
    };
    struct Exit {
      Type type;
      AlignmentState physical;
    };
    DenseMap<Value, Exit> exits;
    auto eraseDead = [&] {
      for (auto &op : llvm::make_early_inc_range(llvm::reverse(func.front())))
        if (!isa<BootstrapOp>(op) && !fixed(&op) &&
            llvm::none_of(op.getResults(), [&](Value v) { return boundaries.contains(v); }) &&
            isOpTriviallyDead(&op)) {
          for (Value v : op.getResults()) exits.erase(v);
          op.erase();
        }
    };
    // Repair explicit unsafe units first. Removal is restricted to private dead
    // operations; shared values remain immutable boundaries.
    SmallVector<MulOp> units;
    GraphPlanner original(waterline, unitError, boundaries, maxGraphQueries, maxGraphDepth);
    func.walk([&](MulOp op) {
      auto index = cipherOperand(op);
      if (!index || !unit(op->getOperand(1-*index).getDefiningOp<ConstantOp>(), true)) return;
      // A precise unit can still be too late: inspect both inputs of private
      // binary nodes for earlier RS states below the floor as well.
      bool needsRepair = !preciseUnit(actualScale(op->getOperand(1-*index)), unitError) ||
                         !original.validOriginal(op->getOperand(*index));
      if (needsRepair) units.push_back(op);
    });
    for (auto op : units) {
      if (fixed(op)) return op.emitError("hoist-scale-alignment: unsafe fixed unit");
      Value old = op.getResult();
      Value source = op->getOperand(*cipherOperand(op));
      auto replacement = apply(source, state(old), op, getScaleType(old).getScale());
      if (failed(replacement)) return failure();
      if (replacement->getType() != old.getType())
        return op.emitError("hoist-scale-alignment: replacement changed nominal exit type");
      exits.try_emplace(*replacement, Exit{old.getType(), state(old)});
      old.replaceAllUsesWith(*replacement);
      for (auto &entry : saved) if (entry.first == old) entry.first = *replacement;
      boundaries.erase(old);
      boundaries.insert(*replacement);
      op.erase();
      clearActualScales(func);
    }
    // A replaced graph can contain old Add nodes. Remove that dead graph before
    // collecting remaining alignment requests, rather than planning it twice.
    eraseDead();
    SmallVector<AddOp> adds;
    func.walk([&](AddOp add) {
      if (getScaleType(add.getLhs()).isCipher() && getScaleType(add.getRhs()).isCipher()) adds.push_back(add);
    });
    for (auto add : adds) {
      double lhs = actualScale(add.getLhs()), rhs = actualScale(add.getRhs());
      if (std::abs(lhs-rhs) <= tolerance) continue;
      if (fixed(add)) return add.emitError("hoist-scale-alignment: fixed Add has incompatible scales");
      unsigned low = lhs < rhs ? 0 : 1;
      Value source = add->getOperand(low);
      if (level(source) != level(add->getOperand(1-low)))
        return add.emitError("hoist-scale-alignment requires an existing common Add level");
      exits.try_emplace(add.getResult(), Exit{add.getType(), state(add.getResult())});
      auto replacement = apply(source, {level(source), std::max(lhs,rhs)}, add,
                               getScaleType(add.getResult()).getScale());
      if (failed(replacement)) return failure();
      add->setOperand(low, *replacement);
      clearActualScales(func);
    }
    // Drop only dead private arithmetic/constants, preserving all Boot nodes.
    eraseDead();
    clearActualScales(func);
    // Restore consistent nominal types; the exact scale remains in encoding
    // attributes and is recomputed from real arithmetic, never overwritten.
    for (auto &op : func.front()) {
      if (isa<ConstantOp>(op) || op.getNumResults() == 0) continue;
      auto inference = dyn_cast<InferTypeOpInterface>(&op);
      SmallVector<Type> types;
      if (!inference || failed(inference.inferReturnTypes(op.getContext(), op.getLoc(), op.getOperands(),
              op.getAttrDictionary(), op.getPropertiesStorage(), op.getRegions(), types)))
        return op.emitError("hoist-scale-alignment: result type inference failed");
      op.getResult(0).setType(types.front());
      auto s = state(op.getResult(0));
      if (auto it = exits.find(op.getResult(0)); it != exits.end()) {
        const auto &expected = it->second;
        if (types.front() != expected.type || s.level != expected.physical.level ||
            std::abs(s.scale - expected.physical.scale) > tolerance)
          return op.emitError("hoist-scale-alignment: rebuilt exit disagrees with original type or physical state")
                 << "; expected type " << expected.type << ", actual type " << types.front()
                 << ", expected physical scale " << expected.physical.scale
                 << ", actual physical scale " << s.scale;
      }
      if (s.level < static_cast<unsigned>(EarthDialect::nativeArithmeticLowerLevel) ||
          !std::isfinite(s.scale) || s.scale < waterline-tolerance ||
          s.scale + 1 >= EarthDialect::nativeScaleChain.capacity(s.level))
        return op.emitError("hoist-scale-alignment: final state violates scale/level capacity");
    }
    for (auto [v, expected] : saved)
      if (level(v) != expected.level || std::abs(actualScale(v)-expected.scale) > tolerance)
        return func.emitError("hoist-scale-alignment: fixed boundary state changed");
    for (auto &op : func.front()) if (auto c = dyn_cast<ConstantOp>(op)) {
      double s = actualScale(c.getResult());
      if (!std::isfinite(s) || s + 1 >= EarthDialect::nativeScaleChain.capacity(level(c.getResult())) ||
          (unit(c) ? !preciseUnit(s, unitError) : s < waterline-tolerance))
        return c.emitError("hoist-scale-alignment: final plaintext encoding precision is invalid");
    }
    auto ret = cast<func::ReturnOp>(func.front().getTerminator());
    func.setFunctionType(builder.getFunctionType(func.getArgumentTypes(), ret.getOperandTypes()));
    func->setAttr("earth.scale_alignment_trace", builder.getArrayAttr(trace));
    func->setAttr("earth.native_scale_moduli", builder.getDenseI64ArrayAttr(EarthDialect::nativeModuli));
    func->setAttr("earth.native_output_level", builder.getI64IntegerAttr(EarthDialect::nativeOutputLevel));
    func->setAttr("earth.native_level_anchor", builder.getI64IntegerAttr(EarthDialect::nativeLevelAnchor));
    func->setAttr("earth.scale_waterline", builder.getF64FloatAttr(waterline));
    return verify(func);
  }
  void runOnOperation() override {
    auto func = getOperation();
    if (func.isExternal()) return;
    if (!usesNativeScalePlanning() || !llvm::hasSingleElement(func.getBody()) ||
        !std::isfinite(waterline) || waterline <= 0 || !std::isfinite(unitError) || unitError < 0 ||
        !maxGraphQueries || !maxGraphDepth) {
      func.emitError("hoist-scale-alignment requires native parameters, one block, and valid precision limits");
      return signalPassFailure();
    }
    auto initial = func->getAttrOfType<IntegerAttr>("init_level");
    if (!initial || initial.getInt() != EarthDialect::nativeLevelAnchor) {
      func.emitError("hoist-scale-alignment: init_level disagrees with native level anchor");
      return signalPassFailure();
    }
    // Transactional: neither a failed search nor a failed final verification
    // leaves partly rewritten input IR in the pass manager.
    OwningOpRef<func::FuncOp> trial(func.clone());
    if (failed(repair(*trial))) return signalPassFailure();
    func->setAttrs((*trial)->getAttrs());
    func.getBody().takeBody(trial->getBody());
  }
};
} // namespace
