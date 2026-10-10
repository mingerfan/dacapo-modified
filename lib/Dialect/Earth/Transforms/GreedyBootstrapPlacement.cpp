#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
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
  }

  void runOnOperation() override {
    auto func = getOperation();
    const int64_t factor = EarthDialect::rescalingFactor;
    const int64_t lastBootLevel = EarthDialect::bootstrapLevelUpperBound -
                                  EarthDialect::bootstrapLevelLowerBound;
    if (func.isExternal())
      return;
    if (!llvm::hasSingleElement(func.getBody()) || waterline <= 0 ||
        waterline > factor || lastBootLevel < 2) {
      func.emitError(
          "greedy bootstrap requires one block, 0 < waterline <= "
          "rescalingFactor, and at least two usable bootstrap levels");
      return signalPassFailure();
    }

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

      SmallVector<uint64_t> cost(maxDepth + 1,
                                 std::numeric_limits<uint64_t>::max());
      SmallVector<int64_t> previous(maxDepth + 1, -1);
      // A bootstrap normalized back to the waterline already consumes one
      // level. Fresh inputs have the full initial budget.
      int64_t initialBudget = lastBootLevel;
      for (auto arg : func.getArguments())
        initialBudget = std::min(initialBudget,
                                 lastBootLevel - getScaleType(arg).getLevel());
      cost[0] = 0;
      for (int64_t end = 1; end <= maxDepth; ++end) {
        if (end <= initialBudget)
          cost[end] = 0;
        for (int64_t start = end - 1;
             start >= std::max<int64_t>(0, end - (lastBootLevel - 1));
             --start) {
          uint64_t candidate = cost[start] + crossing[start];
          if (candidate < cost[end]) {
            cost[end] = candidate;
            previous[end] = start;
          }
        }
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
      int64_t scale = getScaleType(value).getScale();
      if (scale == waterline)
        return value;
      value = builder.create<UpscaleOp>(value.getLoc(), value,
                                        factor + waterline - scale);
      return builder.create<RescaleOp>(value.getLoc(), value);
    };

    uint64_t bootCount = 0, barriers = 0;
    size_t nextCut = 0;
    auto refresh = [&](Value logical) {
      auto value = resolve(logical);
      auto type = getScaleType(value);
      if (type.getScale() != factor)
        value = builder.create<UpscaleOp>(value.getLoc(), value,
                                          factor - type.getScale());
      value = builder.create<BootstrapOp>(value.getLoc(), value);
      current[logical] = normalize(value);
      ++bootCount;
    };
    for (auto *op : operations) {
      SmallVector<Value> logicalOperands(op->getOperands());
      builder.setInsertionPoint(op);
      if (depth_dp && nextCut < cuts.size() &&
          depths.lookup(op) > cuts[nextCut]) {
        ++barriers;
        int64_t end = nextCut + 1 < cuts.size() ? cuts[nextCut + 1] : maxDepth;
        int64_t safeLevel = lastBootLevel - (end - cuts[nextCut]);
        for (auto logical : live) {
          auto value = resolve(logical);
          auto type = getScaleType(value);
          if (type.getLevel() > lastBootLevel || type.getScale() != waterline) {
            op->emitError(
                "live ciphertext exceeds the greedy bootstrap budget");
            return signalPassFailure();
          }
          // A shortcut with enough remaining levels gains nothing from Boot.
          if (type.getLevel() > safeLevel)
            refresh(logical);
        }
        ++nextCut;
      }
      if (isa<MulOp>(op)) {
        bool refreshed = false;
        for (auto logical : logicalOperands) {
          auto type = getScaleType(resolve(logical));
          if (!type.isCipher() || type.getLevel() < lastBootLevel)
            continue;
          if (depth_dp || type.getLevel() > lastBootLevel) {
            op->emitError(
                "multiplication exceeds the bootstrap placement budget");
            return signalPassFailure();
          }
          refresh(logical);
          refreshed = true;
        }
        if (refreshed)
          ++barriers;
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

    refineReturnValues(func, builder, inputTypes, waterline, output_val);
    llvm::outs() << (depth_dp ? "Depth-DP" : "Greedy")
                 << " bootstrap boundaries: " << barriers << '\n'
                 << "Number of Bootstrapping: " << bootCount << '\n';
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<EarthDialect>();
  }
};
} // namespace
