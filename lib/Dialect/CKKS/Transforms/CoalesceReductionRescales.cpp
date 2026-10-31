#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "hecate/Dialect/CKKS/Transforms/Passes.h"
#include "hecate/Support/ModulusChain.h"
#include "mlir/IR/PatternMatch.h"
#include "nlohmann/json.hpp"
#include "llvm/ADT/DenseMap.h"
#include <cmath>
#include <cstdint>
#include <fstream>
#include <utility>
#include <vector>

namespace hecate::ckks {
#define GEN_PASS_DEF_COALESCEREDUCTIONRESCALES
#include "hecate/Dialect/CKKS/Transforms/Passes.h.inc"
} // namespace hecate::ckks

using namespace mlir;
namespace ckks = hecate::ckks;

namespace {
struct Costs {
  SmallVector<double> add, rescalePrefix;

  LogicalResult read(StringRef path, ArrayRef<int64_t> moduli,
                     func::FuncOp func) {
    std::ifstream input(path.str());
    auto spec = nlohmann::json::parse(input, nullptr, false);
    if (spec.is_discarded() || !spec.is_object() || !spec.contains("context") ||
        !spec["context"].contains("rns_moduli") ||
        spec["context"]["rns_moduli"] !=
            std::vector<int64_t>(moduli.begin(), moduli.end()) ||
        !spec.contains("operators"))
      return func.emitError("reduction Rescale requires an OperatorSpec "
                            "matching the actual Q chain");
    SmallVector<double> rescale;
    for (auto entry : {std::make_pair("add_cc", &add),
                       std::make_pair("rescale", &rescale)}) {
      const auto &operators = spec["operators"];
      if (!operators.contains(entry.first) ||
          !operators[entry.first].contains("supported") ||
          operators[entry.first]["supported"] != true ||
          !operators[entry.first].contains("latency_us_by_level") ||
          !operators[entry.first]["latency_us_by_level"].is_array())
        return func.emitError(
            "reduction Rescale requires AddCC and Rescale latency tables");
      for (const auto &value : operators[entry.first]["latency_us_by_level"]) {
        if (!value.is_number() || !std::isfinite(value.get<double>()) ||
            value.get<double>() < 0)
          return func.emitError(
              "reduction Rescale latency must be finite and nonnegative");
        entry.second->push_back(value.get<double>());
      }
    }
    rescalePrefix.push_back(0);
    for (double cost : rescale)
      rescalePrefix.push_back(rescalePrefix.back() + cost);
    return success();
  }

  FailureOr<bool> profitable(unsigned source, unsigned target,
                             Operation *op) const {
    if (source >= add.size() || source + 1 >= rescalePrefix.size())
      return op->emitError(
                 "reduction Rescale latency tables do not cover the source Q"),
             failure();
    double saved = rescalePrefix[source + 1] - rescalePrefix[target + 1];
    if (add[source] <= 0 || add[target] <= 0 || saved <= 0)
      return op->emitError("reduction Rescale requires positive latency at the "
                           "used Q levels"),
             failure();
    return saved > add[source] - add[target];
  }
};

// An entry represents either one private RS chain after MulCP, or an AddCC
// tree of such chains. Each entry has constant size, including for long sums.
struct Reduction {
  Value raw, exemplar;
  Type rawType;
  double rawScale;
  uint64_t terms;
  bool child = false;
};

struct CoalesceReductionRescalesPass
    : ckks::impl::CoalesceReductionRescalesBase<CoalesceReductionRescalesPass> {
  using Base::Base;

  void runOnOperation() override {
    auto func = getOperation();
    auto moduli = func->getAttrOfType<DenseI64ArrayAttr>("ckks.rns_moduli");
    if (func.isExternal() || !moduli)
      return; // Lazy/eager plans do not describe native scale transitions.
    if (!func.getBody().hasOneBlock() || func->hasAttr("dist.device_counts")) {
      func.emitError(
          "reduction Rescale requires one unplaced physical CKKS block");
      return signalPassFailure();
    }
    for (auto q : moduli.asArrayRef())
      if (q <= 1 || q >= (int64_t{1} << 53)) {
        func.emitError("reduction Rescale requires moduli in (1, 2^53)");
        return signalPassFailure();
      }
    hecate::ModulusChain chain;
    chain.reset(moduli.asArrayRef());
    llvm::DenseMap<Value, Reduction> reductions;
    Costs costs;
    bool haveCosts = false;
    auto scaleOf = [](Value value) -> FailureOr<double> {
      auto *op = value.getDefiningOp();
      auto scale = op->getAttrOfType<FloatAttr>("runtime.scale_log2");
      if (!scale || !std::isfinite(scale.getValueAsDouble()) ||
          scale.getValueAsDouble() < 0 || scale.getValueAsDouble() > 1023)
        return op->emitError(
                   "reduction Rescale requires materialized actual scales"),
               failure();
      return scale.getValueAsDouble();
    };

    // Analyze without mutating. Consecutive RS summaries and addition summaries
    // are extended in O(1), rather than rescanning every partial sum's history.
    for (auto &op : func.front()) {
      if (op.hasAttr("sm_fixed"))
        continue;
      if (auto rs = dyn_cast<ckks::RescaleCOp>(op)) {
        Value input = rs.getSrc();
        auto source = ckks::getPolyType(input),
             target = ckks::getPolyType(rs.getResult());
        if (source.getLevel() != target.getLevel() + 1)
          continue;
        if (auto mul = input.getDefiningOp<ckks::MulCPOp>();
            mul && !mul->hasAttr("sm_fixed")) {
          auto scale = scaleOf(input);
          if (failed(scale))
            return signalPassFailure();
          reductions[rs.getResult()] = {input, rs.getResult(), input.getType(),
                                        *scale, 1};
        } else if (input.hasOneUse() &&
                   input.getDefiningOp<ckks::RescaleCOp>()) {
          auto previous = reductions.find(input);
          if (previous != reductions.end()) {
            auto next = previous->second;
            next.exemplar = rs.getResult();
            reductions[rs.getResult()] = next;
          }
        }
        continue;
      }
      auto add = dyn_cast<ckks::AddCCOp>(op);
      if (!add || !add.getLhs().hasOneUse() || !add.getRhs().hasOneUse())
        continue;
      auto left = reductions.find(add.getLhs()),
           right = reductions.find(add.getRhs());
      if (left == reductions.end() || right == reductions.end())
        continue;
      auto a = left->second, b = right->second;
      if (a.rawType != b.rawType || a.rawScale != b.rawScale ||
          add.getLhs().getType() != add.getRhs().getType())
        continue;
      unsigned source = cast<ckks::PolyTypeInterface>(
                            cast<RankedTensorType>(a.rawType).getElementType())
                            .getLevel();
      unsigned target = ckks::getPolyType(add.getResult()).getLevel();
      uint64_t terms = a.terms + b.terms;
      // Reserve carry bits for every term, including every partial sum. This
      // extends the existing scale-capacity check, not a full range analysis.
      if (source >= chain.size() ||
          a.rawScale + std::log2(static_cast<double>(terms)) >=
              chain.capacity(source))
        continue;
      if (!haveCosts) {
        if (failed(costs.read(operatorSpecPath, moduli.asArrayRef(), func)))
          return signalPassFailure();
        haveCosts = true;
      }
      auto profitable = costs.profitable(source, target, add);
      if (failed(profitable))
        return signalPassFailure();
      if (!*profitable)
        continue;
      // DenseMap insertion can invalidate its iterators, so mark children
      // first.
      left->second.child = right->second.child = true;
      reductions[add.getResult()] = {add.getResult(), a.exemplar, a.rawType,
                                     a.rawScale, terms};
    }

    IRRewriter rewriter(func.getContext());
    SmallVector<Value> roots;
    for (auto &op : func.front())
      if (auto add = dyn_cast<ckks::AddCCOp>(op)) {
        auto entry = reductions.find(add.getResult());
        if (entry != reductions.end() && !entry->second.child)
          roots.push_back(add.getResult());
      }
    for (Value root : roots) {
      const auto &region = reductions.find(root)->second;
      // Snapshot one original RS sequence: copy its exact types and actual
      // scales, so all consumers retain precisely the planned output state.
      SmallVector<ckks::RescaleCOp> pattern;
      Value cursor = region.exemplar;
      while (auto rs = cursor.getDefiningOp<ckks::RescaleCOp>()) {
        pattern.push_back(rs);
        cursor = rs.getSrc();
      }
      rewriter.setInsertionPointAfter(root.getDefiningOp());
      Value output = root;
      Operation *first = nullptr;
      for (auto rs : llvm::reverse(pattern)) {
        auto replacement = rewriter.create<ckks::RescaleCOp>(
            rs.getLoc(), rs.getResult().getType(), output);
        auto scale = rs->getAttrOfType<FloatAttr>("runtime.scale_log2");
        if (!scale) {
          rs.emitError("reduction Rescale requires materialized actual scales");
          return signalPassFailure();
        }
        replacement->setAttr("runtime.scale_log2", scale);
        if (!first)
          first = replacement;
        output = replacement.getResult();
      }
      rewriter.replaceAllUsesExcept(root, output, first);

      SmallVector<Value> pending{root}, leaves;
      while (!pending.empty()) {
        Value value = pending.pop_back_val();
        const auto &entry = reductions.find(value)->second;
        if (auto add = value.getDefiningOp<ckks::AddCCOp>()) {
          for (unsigned i = 0; i < 2; ++i) {
            Value operand = add->getOperand(i);
            pending.push_back(operand);
            add->setOperand(i, reductions.find(operand)->second.raw);
          }
          value.setType(entry.rawType);
          add->setAttr("runtime.scale_log2",
                       rewriter.getF64FloatAttr(entry.rawScale));
        } else {
          leaves.push_back(value);
        }
      }
      for (Value leaf : leaves) {
        Value raw = reductions.find(leaf)->second.raw;
        while (leaf != raw) {
          auto rs = leaf.getDefiningOp<ckks::RescaleCOp>();
          Value input = rs.getSrc();
          assert(rs->use_empty() && "only private RS chains may be removed");
          rewriter.eraseOp(rs);
          leaf = input;
        }
      }
    }
  }
};
} // namespace
