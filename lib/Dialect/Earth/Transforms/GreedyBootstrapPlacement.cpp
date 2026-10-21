#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <iterator>
#include <limits>

namespace hecate::earth {
#define GEN_PASS_DEF_GREEDYBOOTSTRAPPLACEMENT
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace hecate::earth

using namespace mlir;
using namespace hecate::earth;

namespace {
struct GreedyBootstrapPlacementPass
    : hecate::earth::impl::GreedyBootstrapPlacementBase<
          GreedyBootstrapPlacementPass> {
  GreedyBootstrapPlacementPass() = default;
  GreedyBootstrapPlacementPass(GreedyBootstrapPlacementOptions options) {
    waterline = options.waterline;
    output_val = options.output_val;
    depth_dp = options.depth_dp;
    bootstrap_scale = options.bootstrap_scale;
  }

  void runOnOperation() override {
    auto func = getOperation();
    const int64_t factor = EarthDialect::rescalingFactor;
    const int64_t bootScale = bootstrap_scale ? bootstrap_scale.getValue() : factor;
    const int64_t bootNormalizationLevels =
        factor > 0 && bootScale >= waterline ? (bootScale - waterline) / factor : 0;
    const int64_t lastBootLevel = EarthDialect::bootstrapLevelUpperBound -
                                  EarthDialect::bootstrapLevelLowerBound;
    if (func.isExternal())
      return;
    if (!llvm::hasSingleElement(func.getBody()) || waterline <= 0 ||
        factor <= 0 || bootScale < waterline ||
        lastBootLevel <= bootNormalizationLevels) {
      func.emitError(
          "greedy bootstrap requires one block, positive scales/rescalingFactor, "
          "Boot scale >= waterline and a usable bootstrap budget");
      return signalPassFailure();
    }

    // Dead branches must not create depth cuts that refresh live outputs.
    // Reverse order removes a whole dead def-use chain in one traversal.
    for (auto &op : llvm::make_early_inc_range(llvm::reverse(func.front())))
      if (isOpTriviallyDead(&op))
        op.erase();

    // Keep the original use counts: the management operations inserted below
    // must not extend the lifetime of a logical value in this scan.
    SmallVector<Operation *> operations;
    llvm::DenseMap<Value, unsigned> remainingUses;
    llvm::DenseMap<Value, Value> current;
    llvm::SetVector<Value> live;
    llvm::DenseMap<Operation *, int64_t> depths;
    llvm::DenseMap<Value, int64_t> valueDepths;
    for (auto arg : func.getArguments()) {
      remainingUses[arg] = std::distance(arg.use_begin(), arg.use_end());
      if (remainingUses[arg])
        live.insert(arg);
    }
    for (auto &op : func.front()) {
      if (!isa<ConstantOp, MulOp, AddOp, RotateOp, NegateOp, func::ReturnOp>(
              op)) {
        op.emitError("unsupported operation in greedy bootstrap input");
        return signalPassFailure();
      }
      operations.push_back(&op);
      int64_t depth = 0;
      for (auto operand : op.getOperands())
        depth = std::max(depth, valueDepths.lookup(operand));
      if (isa<MulOp>(op))
        ++depth;
      depths[&op] = depth;
      for (auto result : op.getResults()) {
        remainingUses[result] =
            std::distance(result.use_begin(), result.use_end());
        valueDepths[result] = depth;
      }
    }

    OpBuilder builder(func);
    SmallVector<Type, 4> inputTypes;
    refineInputValues(func, builder, inputTypes, waterline, output_val);
    for (auto arg : func.getArguments()) {
      auto type = getScaleType(arg);
      if (!type.isCipher() || type.getLevel() > lastBootLevel) {
        func.emitError(
            "greedy bootstrap requires bootstrappable ciphertext inputs");
        return signalPassFailure();
      }
    }

    // Fhelipe (PLDI 2024, sections 6.2-6.3) restricts the search to
    // multiplicative-depth boundaries. This implementation minimizes a
    // conservative count of unique ciphertexts crossing those boundaries.
    // Delay unary operations to their first consumer to narrow fanout frontiers.
    // Keep additions early so reductions complete before the boundary; delaying
    // a whole reduction would refresh all of its partial products instead.
    // This is a cheap scheduling heuristic, not upstream's chokepoint selection.
    SmallVector<int64_t> cuts;
    int64_t maxDepth = 0;
    for (auto *op : operations)
      maxDepth = std::max(maxDepth, depths.lookup(op));
    if (depth_dp) {
      depths[operations.back()] = maxDepth + 1;
      for (auto *op : llvm::reverse(operations)) {
        if (!isa<RotateOp, NegateOp>(op) ||
            op->getResult(0).use_empty())
          continue;
        int64_t latest = maxDepth + 1;
        for (auto *user : op->getResult(0).getUsers())
          latest = std::min(latest, depths.lookup(user));
        depths[op] = latest;
      }
      SmallVector<int64_t> crossing(maxDepth + 2, 0);
      auto countCrossing = [&](Value value, int64_t start) {
        if (!remainingUses.lookup(value) || !getScaleType(value).isCipher())
          return;
        int64_t end = start;
        for (auto *user : value.getUsers())
          end = std::max(end, depths.lookup(user));
        ++crossing[start];
        --crossing[end];
      };
      for (auto arg : func.getArguments())
        countCrossing(arg, 0);
      for (auto *op : operations)
        for (auto result : op->getResults())
          countCrossing(result, depths.lookup(op));
      for (int64_t i = 1; i <= maxDepth; ++i)
        crossing[i] += crossing[i - 1];

      // Simulate the same EVA scale transitions used by emission. A multiply
      // may need zero, one or several Rescales; multiplication depth alone is
      // not a modulus budget.
      llvm::stable_sort(operations, [&](Operation *a, Operation *b) {
        return depths.lookup(a) < depths.lookup(b);
      });
      struct State { int64_t level, scale; };
      SmallVector<uint64_t> cost(maxDepth + 1,
                                 std::numeric_limits<uint64_t>::max());
      SmallVector<int64_t> previous(maxDepth + 1, -1);
      cost[0] = 0;
      // Also consider refreshing the depth-zero frontier before the first
      // multiply. Inputs can already have consumed the entire Q budget.
      // The virtual start -1 represents this initial refresh; start 0 keeps
      // the original input states.
      for (int64_t start = -1; start < maxDepth; ++start) {
        if (start >= 0 && cost[start] == std::numeric_limits<uint64_t>::max())
          continue;
        const int64_t cut = std::max<int64_t>(start, 0);
        llvm::DenseMap<Value, State> states;
        auto stateOf = [&](Value value) -> State {
          auto it = states.find(value);
          if (it != states.end()) return it->second;
          if (start == 0) {
            auto type = getScaleType(value);
            return {static_cast<int64_t>(type.getLevel()), static_cast<int64_t>(type.getScale())};
          }
          return {bootNormalizationLevels,
                  bootScale - bootNormalizationLevels * factor};
        };
        bool valid = true;
        size_t next = 0;
        while (start != 0 && next < operations.size() && depths.lookup(operations[next]) <= cut)
          ++next;
        for (int64_t end = cut + 1; end <= maxDepth && valid; ++end) {
          while (next < operations.size() && depths.lookup(operations[next]) <= end) {
            auto *op = operations[next++];
            if (isa<ConstantOp, func::ReturnOp>(op)) continue;
            State result{0, 0};
            bool first = true;
            for (Value input : op->getOperands()) {
              if (!getScaleType(input).isCipher()) continue;
              State operand = stateOf(input);
              result.level = std::max(result.level, operand.level);
              if (isa<MulOp>(op)) result.scale += operand.scale;
              else result.scale = std::max(result.scale, operand.scale);
              first = false;
            }
            if (first) { valid = false; break; }
            if (auto mul = dyn_cast<MulOp>(op)) {
              if (mul.isSingle()) result.scale += waterline;
              while (result.scale >= waterline + factor) {
                result.scale -= factor;
                ++result.level;
              }
            }
            if (result.level > lastBootLevel) { valid = false; break; }
            for (Value output : op->getResults()) states[output] = result;
          }
          if (valid) {
            uint64_t candidate = start < 0 ? crossing[0]
                                          : cost[start] + (start ? crossing[start] : 0);
            if (candidate < cost[end]) {
              cost[end] = candidate;
              previous[end] = start < 0 ? 0 : (start ? start : -1);
            }
          }
        }
      }
      if (cost[maxDepth] == std::numeric_limits<uint64_t>::max()) {
        func.emitError("no feasible depth-DP placement for the EVA Q budget");
        return signalPassFailure();
      }
      for (int64_t end = maxDepth; previous[end] >= 0;) {
        end = previous[end];
        cuts.push_back(end);
      }
      std::reverse(cuts.begin(), cuts.end());
      llvm::stable_sort(operations, [&](Operation *a, Operation *b) {
        return depths.lookup(a) < depths.lookup(b);
      });
      // Original order breaks ties, so same-depth def-use chains still
      // dominate.
      for (auto *op : operations)
        op->moveBefore(&func.front(), func.front().end());
    }

    auto resolve = [&](Value value) {
      auto it = current.find(value);
      return it == current.end() ? value : it->second;
    };
    auto normalize = [&](Value value) -> Value {
      return normalizeScale(value, builder, waterline);
    };

    uint64_t bootCount = 0, barriers = 0;
    size_t nextCut = 0;
    auto refresh = [&](Value logical) {
      auto value = resolve(logical);
      value = builder.create<BootstrapOp>(value.getLoc(), value, 0, bootScale);
      current[logical] = normalize(value);
      ++bootCount;
    };
    for (auto *op : operations) {
      SmallVector<Value> logicalOperands(op->getOperands());
      builder.setInsertionPoint(op);
      if (depth_dp && nextCut < cuts.size() &&
          depths.lookup(op) > cuts[nextCut]) {
        ++barriers;
        // The segment simulation refreshes every crossing ciphertext. Keep
        // emission identical: skipping a refresh could change both scale and
        // the number of Rescales in the next segment.
        for (auto logical : live) {
          auto type = getScaleType(resolve(logical));
          if (type.getLevel() > lastBootLevel) {
            op->emitError("live ciphertext exceeds the bootstrap Q budget");
            return signalPassFailure();
          }
          refresh(logical);
        }
        ++nextCut;
      }
      if (isa<MulOp>(op)) {
        auto requiredLevel = [&]() {
          int64_t level = 0, scale = 0;
          for (Value logical : logicalOperands) {
            auto type = getScaleType(resolve(logical));
            if (type.isCipher()) {
              level = std::max(level, static_cast<int64_t>(type.getLevel()));
              scale += type.getScale();
            } else scale += waterline;
          }
          return level + std::max<int64_t>(0, (scale - waterline) / factor);
        };
        if (requiredLevel() > lastBootLevel) {
          if (depth_dp) {
            op->emitError("multiplication exceeds the simulated bootstrap Q budget");
            return signalPassFailure();
          }
          llvm::SetVector<Value> operands;
          for (Value logical : logicalOperands)
            if (getScaleType(resolve(logical)).isCipher()) operands.insert(logical);
          for (Value logical : operands) refresh(logical);
          ++barriers;
          if (requiredLevel() > lastBootLevel) {
            op->emitError("multiplication cannot fit after Boot");
            return signalPassFailure();
          }
        }
      }

      for (auto [index, logical] : llvm::enumerate(logicalOperands)) {
        auto value = resolve(logical);
        // A plaintext's encoding scale/level is local to its consumer.
        if (auto constant = value.getDefiningOp<ConstantOp>();
            constant && !value.hasOneUse())
          value = builder.clone(*constant.getOperation())->getResult(0);
        op->setOperand(index, value);
      }
      if (auto forward = dyn_cast<ForwardMgmtInterface>(op)) {
        forward.processOperandsEVA(waterline);
        auto inference = cast<InferTypeOpInterface>(op);
        SmallVector<Type> types;
        if (failed(inference.inferReturnTypes(
                op->getContext(), op->getLoc(), op->getOperands(),
                op->getAttrDictionary(), op->getPropertiesStorage(),
                op->getRegions(), types))) {
          op->emitError("type inference failed in greedy bootstrap placement");
          return signalPassFailure();
        }
        op->getResult(0).setType(types.front());
        if (isa<MulOp>(op)) {
          builder.setInsertionPointAfter(op);
          current[op->getResult(0)] = normalize(op->getResult(0));
        }
      }
      for (auto logical : logicalOperands) {
        if (--remainingUses[logical] == 0) {
          live.remove(logical);
          current.erase(logical);
        }
      }
      for (auto result : op->getResults())
        if (remainingUses[result] && getScaleType(result).isCipher())
          live.insert(result);
    }

    finishThresholdScale(func, builder, inputTypes);
    llvm::outs() << (depth_dp ? "Depth-DP" : "Greedy")
                 << " bootstrap boundaries: " << barriers << '\n'
                 << "Number of Bootstrapping: " << bootCount << '\n';
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<EarthDialect>();
  }
};
} // namespace
