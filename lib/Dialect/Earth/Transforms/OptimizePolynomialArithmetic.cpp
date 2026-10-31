#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace hecate::earth {
#define GEN_PASS_DEF_OPTIMIZEPOLYNOMIALARITHMETIC
#define GEN_PASS_DEF_ABSORBCOEFFICIENTSCALE
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace hecate::earth

using namespace mlir;
using namespace hecate::earth;

namespace {
struct OptimizePolynomialArithmeticPass
    : hecate::earth::impl::OptimizePolynomialArithmeticBase<
          OptimizePolynomialArithmeticPass> {
  OptimizePolynomialArithmeticPass() = default;
  OptimizePolynomialArithmeticPass(
      OptimizePolynomialArithmeticOptions options) {
    absorb_alignment = options.absorb_alignment;
  }

  void runOnOperation() override {
    auto func = getOperation();
    if (func.isExternal() || func->hasAttr("init_level"))
      return;
    if (absorb_alignment)
      func->setAttr("absorb_coefficient_scale",
                    UnitAttr::get(func.getContext()));
    IRRewriter rewriter(func.getContext());
    func.walk([&](MulOp mul) {
      if (mul->hasAttr("sm_fixed"))
        return;
      Value cipher, plain;
      for (Value operand : mul->getOperands()) {
        if (getScaleType(operand).isCipher())
          cipher = operand;
        else
          plain = operand;
      }
      if (!cipher || !plain)
        return;
      auto constant = plain.getDefiningOp<ConstantOp>();
      if (!constant || constant->hasAttr("sm_fixed"))
        return;
      auto payload = dyn_cast<DenseFPElementsAttr>(constant.getValue());
      // Runtime zero-pads a short payload. It must never become a full-slot
      // doubling.
      if (!payload || !payload.isSplat() ||
          payload.getNumElements() != EarthDialect::polynomialDegree / 2)
        return;
      double coefficient = payload.getSplatValue<APFloat>().convertToDouble();
      if (coefficient == 2.0 && getScaleType(cipher).getScale() == 0 &&
          getScaleType(plain).getScale() == 0) {
        rewriter.setInsertionPoint(mul);
        rewriter.replaceOpWithNewOp<AddOp>(mul, cipher, cipher);
      }
    });
    for (auto &op : llvm::make_early_inc_range(llvm::reverse(func.front())))
      if (isOpTriviallyDead(&op))
        op.erase();
  }
};

struct AbsorbCoefficientScalePass
    : hecate::earth::impl::AbsorbCoefficientScaleBase<
          AbsorbCoefficientScalePass> {
  AbsorbCoefficientScalePass() = default;
  AbsorbCoefficientScalePass(AbsorbCoefficientScaleOptions options) {
    waterline = options.waterline;
  }
  void runOnOperation() override {
    auto func = getOperation();
    if (func->hasAttr("earth.region_scale_policy") || !func->hasAttr("absorb_coefficient_scale"))
      return;
    IRRewriter rewriter(func.getContext());
    func.walk([&](UpscaleOp upscale) {
      if (upscale->hasAttr("sm_fixed") || upscale.getUpFactor() <= 0)
        return;
      Value source = upscale.getValue();
      SmallVector<Operation *> chain;
      while (auto *op = source.getDefiningOp()) {
        if (!isa<RescaleOp, ModswitchOp>(op))
          break;
        if (!source.hasOneUse() || op->hasAttr("sm_fixed"))
          return;
        chain.push_back(op);
        source = op->getOperand(0);
      }
      auto mul = source.getDefiningOp<MulOp>();
      if (!mul || !source.hasOneUse() || mul->hasAttr("sm_fixed"))
        return;
      unsigned index = getScaleType(mul.getLhs()).isCipher() ? 1 : 0;
      Value plain = mul->getOperand(index), payloadSource = plain;
      // EarlyModswitch can leave level-only wrappers around the coefficient.
      // Clone its payload at the effective consumer level, without changing
      // that shared wrapper or its other users.
      while (auto drop = payloadSource.getDefiningOp<ModswitchOp>()) {
        if (drop->hasAttr("sm_fixed"))
          return;
        payloadSource = drop.getValue();
      }
      auto constant = payloadSource.getDefiningOp<ConstantOp>();
      if (!constant || constant->hasAttr("sm_fixed") ||
          !getScaleType(mul->getOperand(1 - index)).isCipher())
        return;
      int64_t increase = upscale.getUpFactor();
      // Raising a normalized result below the threshold preserves exactly the
      // original number of immediate Rescales. Do not silently defer a drop.
      auto result = upscale.getResult();
      auto normalized = normalizeScaleState(
          {static_cast<int64_t>(getScaleType(result).getLevel()), actualScale(result)}, waterline);
      if (normalized.level > getScaleType(result).getLevel())
        return;
      rewriter.setInsertionPoint(mul);
      auto copy = rewriter.create<ConstantOp>(
          constant.getLoc(),
          getScaleType(plain).switchScale(getScaleType(plain).getScale() +
                                          increase),
          constant.getValue(), constant.getRmsVar());
      mul->setOperand(index, copy);
      Value product = mul.getResult();
      product.setType(getScaleType(product).switchScale(
          getScaleType(product).getScale() + increase));
      for (Operation *op : chain) {
        Value result = op->getResult(0);
        result.setType(getScaleType(result).switchScale(
            getScaleType(result).getScale() + increase));
      }
      rewriter.replaceOp(upscale, upscale.getValue());
      if (usesNativeScalePlanning()) clearActualScales(func);
    });
  }
};
} // namespace
