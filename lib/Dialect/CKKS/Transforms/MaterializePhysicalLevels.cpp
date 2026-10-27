#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/CKKS/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "nlohmann/json.hpp"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <cmath>
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/MathExtras.h"
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace hecate {
namespace ckks {
#define GEN_PASS_DEF_MATERIALIZEPHYSICALLEVELS
#include "hecate/Dialect/CKKS/Transforms/Passes.h.inc"
} // namespace ckks
} // namespace hecate

using namespace mlir;
namespace ckks = hecate::ckks;

namespace {

using Json = nlohmann::json;

struct PhysicalLevelSpec {
  uint64_t lowerBound = 0;
  uint64_t upperBound = 0;
  uint64_t maxRescaleLevels = 0;
  std::vector<uint64_t> modulusBits;
  std::vector<int64_t> moduli;
  bool nativeRescale = false;
  std::optional<int64_t> bootOutputScale;
  std::optional<int64_t> bootInputScale;
  uint64_t bootInputLevel = 0;
};

FailureOr<PhysicalLevelSpec> readPhysicalLevelSpec(llvm::StringRef path,
                                                   func::FuncOp func,
                                                   uint64_t levelFactor,
                                                   llvm::StringRef bootProfile) {
  if (path.empty()) {
    func.emitError("materialize-ckks-physical-levels requires --operator-spec");
    return failure();
  }

  std::ifstream input(path.str(), std::ios::binary);
  if (!input) {
    func.emitError("cannot open physical-level OperatorSpec: ") << path;
    return failure();
  }
  Json spec = Json::parse(input, nullptr, false);
  if (spec.is_discarded() || !spec.is_object()) {
    func.emitError("physical-level OperatorSpec is not valid JSON");
    return failure();
  }

  auto version = spec.find("spec_format_version");
  if (version == spec.end() || !version->is_number_integer() ||
      (version->get<int64_t>() != 1 && version->get<int64_t>() != 2)) {
    func.emitError(
        "physical-level materialization requires OperatorSpec V1 or V2");
    return failure();
  }
  auto status = spec.find("status");
  if (status == spec.end() || !status->is_string() ||
      status->get<std::string>() != "validated") {
    func.emitError(
        "physical-level materialization requires a validated OperatorSpec");
    return failure();
  }
  auto mode = spec.find("rescale_mode");
  if (mode == spec.end() || !mode->is_string() ||
      (mode->get<std::string>() != "lazy" && mode->get<std::string>() != "native")) {
    func.emitError("physical-level materialization requires rescale_mode=lazy");
    return failure();
  }

  auto levels = spec.find("levels");
  if (levels == spec.end() || !levels->is_object()) {
    func.emitError("physical-level OperatorSpec is missing levels");
    return failure();
  }
  auto lower = levels->find("lower_bound");
  auto upper = levels->find("upper_bound");
  if (lower == levels->end() || upper == levels->end() ||
      !lower->is_number_integer() || !upper->is_number_integer()) {
    func.emitError("physical-level OperatorSpec has invalid level bounds");
    return failure();
  }
  const int64_t signedLower = lower->get<int64_t>();
  const int64_t signedUpper = upper->get<int64_t>();
  if (signedLower < 0 || signedUpper < signedLower ||
      static_cast<uint64_t>(signedUpper) >
          std::numeric_limits<unsigned>::max()) {
    func.emitError("physical-level OperatorSpec level bounds are out of range");
    return failure();
  }

  auto context = spec.find("context");
  if (context == spec.end() || !context->is_object()) {
    func.emitError("physical-level OperatorSpec is missing context");
    return failure();
  }
  auto moduli = context->find("rns_moduli_log2");
  if (moduli == context->end() || !moduli->is_array() ||
      moduli->size() <= static_cast<size_t>(signedUpper)) {
    func.emitError(
        "physical-level OperatorSpec modulus chain does not cover upper_bound");
    return failure();
  }

  PhysicalLevelSpec result;
  result.nativeRescale = mode->get<std::string>() == "native";
  result.lowerBound = static_cast<uint64_t>(signedLower);
  result.upperBound = static_cast<uint64_t>(signedUpper);
  if (!bootProfile.empty()) {
    const Json *selected = nullptr;
    if (spec.contains("boot_profiles") && spec["boot_profiles"].is_array())
      for (const auto &profile : spec["boot_profiles"])
        if (profile.value("profile_id", std::string()) == bootProfile.str()) {
          if (selected) {
            func.emitError("duplicate physical-level Boot profile");
            return failure();
          }
          selected = &profile;
        }
    if (!selected || !selected->contains("output_level") ||
        !(*selected)["output_level"].is_number_integer() ||
        !selected->contains("output_scale_log2") ||
        !(*selected)["output_scale_log2"].is_number_integer()) {
      func.emitError("physical-level Boot profile is missing or invalid");
      return failure();
    }
    const auto anchor = (*selected)["output_level"].get<int64_t>();
    if (anchor < signedLower || anchor > signedUpper) {
      func.emitError("Boot output level is outside the physical chain");
      return failure();
    }
    // Use the same recoverable working chain for inputs and refreshed values.
    // Context primes above this anchor remain available inside native Boot.
    result.upperBound = static_cast<uint64_t>(anchor);
    result.bootOutputScale = (*selected)["output_scale_log2"].get<int64_t>();
    if (*result.bootOutputScale <= 0) {
      func.emitError("Boot output scale must be positive");
      return failure();
    }
    if (selected->contains("input_scale_log2")) {
      if (!(*selected)["input_scale_log2"].is_number_integer() ||
          (*selected)["input_scale_log2"].get<int64_t>() <= 0 ||
          !selected->contains("input_level_min") ||
          !(*selected)["input_level_min"].is_number_integer() ||
          !selected->contains("input_level_max") ||
          !(*selected)["input_level_max"].is_number_integer()) {
        func.emitError("native Boot input level/scale contract is invalid");
        return failure();
      }
      auto inputLevel = (*selected)["input_level_min"].get<int64_t>();
      auto inputMax = (*selected)["input_level_max"].get<int64_t>();
      if (!result.nativeRescale || inputLevel < signedLower ||
          inputMax < inputLevel || inputMax > signedUpper || inputLevel + 2 > anchor) {
        func.emitError("native Boot input alignment requires two Q above its input level");
        return failure();
      }
      result.bootInputScale = (*selected)["input_scale_log2"].get<int64_t>();
      result.bootInputLevel = static_cast<uint64_t>(inputLevel);
    }
  }
  result.modulusBits.reserve(moduli->size());
  for (const Json &bits : *moduli) {
    if (!bits.is_number_integer()) {
      func.emitError(
          "physical-level OperatorSpec modulus bit widths must be integers");
      return failure();
    }
    const int64_t value = bits.get<int64_t>();
    if (value <= 0 || value > 64) {
      func.emitError(
          "physical-level OperatorSpec modulus bit widths must be in [1, 64]");
      return failure();
    }
    result.modulusBits.push_back(static_cast<uint64_t>(value));
  }

  if (result.nativeRescale) {
    auto actual = context->find("rns_moduli");
    if (actual == context->end() || !actual->is_array() || actual->size() != moduli->size()) {
      func.emitError("native Rescale requires the actual rns_moduli chain");
      return failure();
    }
    for (const auto &[i, q] : llvm::enumerate(*actual)) {
      if (!q.is_number_integer() || q.get<int64_t>() <= 1 || q.get<int64_t>() >= (int64_t{1} << 53)) {
        func.emitError("native Rescale moduli must be integers in (1, 2^53)");
        return failure();
      }
      result.moduli.push_back(q.get<int64_t>());
      if (llvm::Log2_64(q.get<uint64_t>()) + 1 != result.modulusBits[i]) {
        func.emitError("native Rescale modulus value does not match its declared bit width");
        return failure();
      }
    }
    if (levelFactor != 1) {
      func.emitError("native Rescale requires one physical Q per logical level");
      return failure();
    }
  }

  auto operators = spec.find("operators");
  if (operators == spec.end() || !operators->is_object()) {
    func.emitError("physical-level OperatorSpec is missing operators");
    return failure();
  }
  auto rescale = operators->find("rescale");
  if (rescale == operators->end() || !rescale->is_object()) {
    func.emitError("physical-level OperatorSpec is missing Rescale support");
    return failure();
  }
  auto supported = rescale->find("supported");
  auto maxLevels = rescale->find("max_levels_per_op");
  if (supported == rescale->end() || !supported->is_boolean() ||
      !supported->get<bool>() || maxLevels == rescale->end() ||
      !maxLevels->is_number_integer()) {
    func.emitError("physical-level OperatorSpec does not support Rescale");
    return failure();
  }
  const int64_t signedMaxLevels = maxLevels->get<int64_t>();
  if (signedMaxLevels <= 0 ||
      static_cast<uint64_t>(signedMaxLevels) < levelFactor) {
    func.emitError("physical-level OperatorSpec max_levels_per_op is smaller "
                   "than levels-per-logical-level");
    return failure();
  }
  result.maxRescaleLevels = static_cast<uint64_t>(signedMaxLevels);
  return result;
}

LogicalResult materializeActualScales(func::FuncOp func, const PhysicalLevelSpec &spec) {
  llvm::DenseMap<Value, double> scales;
  auto arguments = func->getAttrOfType<DenseF64ArrayAttr>("earth.arg_scales");
  for (auto [i, arg] : llvm::enumerate(func.getArguments()))
    scales[arg] = arguments ? arguments.asArrayRef()[i] : ckks::getPolyType(arg).getScaleLog2();
  OpBuilder builder(func.getContext());
  auto annotate = [&](Operation *op, double scale) {
    op->setAttr("runtime.scale_log2", builder.getF64FloatAttr(scale));
    scales[op->getResult(0)] = scale;
  };
  for (auto &operation : llvm::make_early_inc_range(func.front())) {
    auto *op = &operation;
    if (op->getNumResults() == 0) continue;
    if (auto encode = dyn_cast<ckks::EncodeOp>(op)) {
      annotate(op, ckks::getPolyType(encode.getResult()).getScaleLog2());
      continue;
    }
    if (auto boot = dyn_cast<ckks::BootstrapCOp>(op)) {
      if (spec.bootInputScale) {
        Value input = boot->getOperand(0);
        auto inputType = cast<RankedTensorType>(input.getType());
        auto poly = cast<ckks::PolyTypeInterface>(inputType.getElementType());
        const auto targetLevel = spec.bootInputLevel;
        const double targetScale = *spec.bootInputScale;
        double scale = scales.lookup(input);
        if (poly.getLevel() < targetLevel)
          return op->emitError("native Boot input is below the required level");
        unsigned drops = 0;
        double plainScale = targetScale - scale;
        if (std::abs(scale - targetScale) > 1e-8) {
          // Prefer one Q, but keep enough encoding precision for the unit.
          do {
            ++drops;
            if (targetLevel + drops > poly.getLevel())
              return op->emitError("native Boot scale alignment has insufficient Q/encoding precision");
            plainScale += std::log2(static_cast<double>(spec.moduli.at(targetLevel + drops)));
          } while (plainScale < 30);
        }
        builder.setInsertionPoint(op);
        auto typeAt = [&](unsigned level, double actual, unsigned components) {
          auto element = poly.switchLevel(level)
              .switchScaleLog2(static_cast<int64_t>(std::ceil(actual)))
              .switchComponents(components);
          return RankedTensorType::get(inputType.getShape(), element);
        };
        if (poly.getLevel() > targetLevel + drops) {
          auto mod = builder.create<ckks::ModswitchCOp>(op->getLoc(),
              typeAt(targetLevel + drops, scale, poly.getComponents()), input,
              poly.getLevel() - targetLevel - drops);
          annotate(mod, scale);
          input = mod.getResult();
        }
        if (drops) {
          double capacity = 0;
          for (unsigned i = 0; i <= targetLevel + drops; ++i)
            capacity += std::log2(static_cast<double>(spec.moduli.at(i)));
          if (!std::isfinite(plainScale) || scale + plainScale >= capacity)
            return op->emitError("native Boot alignment exceeds the modulus capacity");
          auto payloadType = RankedTensorType::get(
              {hecate::earth::EarthDialect::polynomialDegree / 2}, builder.getF64Type());
          auto one = DenseElementsAttr::get(payloadType, builder.getF64FloatAttr(1.0));
          auto encoded = builder.create<ckks::EncodeOp>(op->getLoc(),
              typeAt(targetLevel + drops, plainScale, 1), one);
          annotate(encoded, plainScale);
          scale += plainScale;
          auto multiplied = builder.create<ckks::MulCPOp>(op->getLoc(),
              typeAt(targetLevel + drops,
                  ckks::getPolyType(input).getScaleLog2() + ckks::getPolyType(encoded.getResult()).getScaleLog2(),
                  poly.getComponents()), input, encoded);
          annotate(multiplied, scale);
          input = multiplied.getResult();
          for (unsigned remaining = drops; remaining > 0; --remaining) {
            scale -= std::log2(static_cast<double>(spec.moduli.at(targetLevel + remaining)));
            auto rescale = builder.create<ckks::RescaleCOp>(op->getLoc(),
                typeAt(targetLevel + remaining - 1, scale, poly.getComponents()), input);
            annotate(rescale, scale);
            input = rescale.getResult();
          }
        }
        if (std::abs(scale - targetScale) > 1e-8)
          return op->emitError("native Boot input scale alignment failed");
        boot->setOperand(0, input);
      }
      annotate(op, ckks::getPolyType(boot.getResult()).getScaleLog2());
      continue;
    }
    if (op->getNumOperands() == 0 || !scales.count(op->getOperand(0)))
      return op->emitError("cannot infer actual CKKS scale");
    double scale = scales.lookup(op->getOperand(0));
    if (isa<ckks::MulCCOp, ckks::MulCPOp>(op)) {
      scale += scales.lookup(op->getOperand(1));
    } else if (auto rescale = dyn_cast<ckks::RescaleCOp>(op)) {
      auto source = ckks::getPolyType(rescale.getSrc());
      auto target = ckks::getPolyType(rescale.getResult());
      for (unsigned level = source.getLevel(); level > target.getLevel(); --level)
        scale -= std::log2(static_cast<double>(spec.moduli.at(level)));
      if (auto floor = func->getAttrOfType<FloatAttr>("earth.scale_waterline");
          floor && scale < floor.getValueAsDouble())
        return op->emitError("native Rescale violates the planned scale waterline");
    } else if (auto add = dyn_cast<ckks::AddCPOp>(op)) {
      auto encode = add.getRhs().getDefiningOp<ckks::EncodeOp>();
      if (!encode) return op->emitError("AddCP requires a local encoding for actual-scale alignment");
      builder.setInsertionPoint(op);
      auto local = builder.create<ckks::EncodeOp>(op->getLoc(), encode.getType(), encode.getPayload());
      annotate(local, scale);
      add.setOperand(1, local.getResult());
      if (encode->use_empty()) encode.erase();
    } else if (auto add = dyn_cast<ckks::AddCCOp>(op)) {
      double rhs = scales.lookup(add.getRhs());
      if (std::abs(scale - rhs) > 1e-10) {
        unsigned lower = scale < rhs ? 0 : 1;
        Value input = op->getOperand(lower);
        double target = std::max(scale, rhs);
        // Reuse an existing encoded-one alignment when it is local to this
        // addition. This preserves its encoding precision and avoids a second
        // near-unit encoding for the actual-prime correction.
        SmallVector<Operation *> chain;
        Value source = input;
        while (auto mod = source.getDefiningOp<ckks::ModswitchCOp>()) {
          if (!source.hasOneUse()) break;
          chain.push_back(mod);
          source = mod.getSrc();
        }
        auto mul = source.getDefiningOp<ckks::MulCPOp>();
        auto unit = mul ? mul.getRhs().getDefiningOp<ckks::EncodeOp>() : ckks::EncodeOp{};
        auto dense = unit ? dyn_cast<DenseElementsAttr>(unit.getPayload()) : DenseElementsAttr{};
        bool reuse = mul && source.hasOneUse() && dense && dense.isSplat() &&
                     dense.getElementType().isF64() && dense.getSplatValue<APFloat>().convertToDouble() == 1.0 &&
                     target >= scales.lookup(mul.getLhs());
        if (reuse) {
          builder.setInsertionPoint(mul);
          auto replacement = builder.create<ckks::EncodeOp>(mul.getLoc(), unit.getType(), unit.getPayload());
          annotate(replacement, target - scales.lookup(mul.getLhs()));
          mul.setOperand(1, replacement.getResult());
          annotate(mul, target);
          for (Operation *forward : chain) annotate(forward, target);
          if (unit->use_empty()) unit.erase();
        } else {
          builder.setInsertionPoint(op);
          auto inputType = cast<RankedTensorType>(input.getType());
          auto poly = inputType.getElementType().cast<ckks::PolyTypeInterface>().switchComponents(1).switchScaleLog2(0);
          auto plainType = RankedTensorType::get(inputType.getShape(), poly);
          auto payloadType = RankedTensorType::get({hecate::earth::EarthDialect::polynomialDegree / 2}, builder.getF64Type());
          auto payload = DenseElementsAttr::get(payloadType, builder.getF64FloatAttr(1.0));
          auto encoded = builder.create<ckks::EncodeOp>(op->getLoc(), plainType, payload);
          annotate(encoded, target - scales.lookup(input));
          auto aligned = builder.create<ckks::MulCPOp>(op->getLoc(), input.getType(), input, encoded);
          annotate(aligned, target);
          op->setOperand(lower, aligned);
        }
        scale = target;
      }
    }
    if (!std::isfinite(scale) || scale < 0 || scale > 1023)
      return op->emitError("actual CKKS scale is outside the supported range");
    if (func->hasAttr("earth.native_scale_moduli") &&
        ckks::getPolyType(op->getResult(0)).getComponents() > 1) {
      double capacity = 0;
      for (unsigned i = 0; i <= ckks::getPolyType(op->getResult(0)).getLevel(); ++i)
        capacity += std::log2(static_cast<double>(spec.moduli[i]));
      if (scale >= capacity)
        return op->emitError("actual CKKS scale exceeds the remaining modulus capacity");
    }
    annotate(op, scale);
  }
  func->setAttr("ckks.rns_moduli", builder.getDenseI64ArrayAttr(spec.moduli));
  if (auto planned = func->getAttrOfType<DenseF64ArrayAttr>("earth.result_scales")) {
    auto outputs = func.front().getTerminator()->getOperands();
    if (planned.size() != outputs.size())
      return func.emitError("native planned result scale count disagrees with function results");
    for (auto [i, output] : llvm::enumerate(outputs))
      if (std::abs(scales.lookup(output) - planned.asArrayRef()[i]) > 1e-8)
        return func.emitError("native planned result scale disagrees with physical replay");
  }
  return success();
}

FailureOr<unsigned> mapLevel(uint64_t logicalLevel, uint64_t logicalBase,
                             const PhysicalLevelSpec &spec,
                             uint64_t levelFactor, Operation *op) {
  if (logicalLevel > logicalBase) {
    op->emitError("logical CKKS level exceeds function init_level");
    return failure();
  }
  const uint64_t consumedLogicalLevels = logicalBase - logicalLevel;
  if (consumedLogicalLevels >
      (spec.upperBound - spec.lowerBound) / levelFactor) {
    op->emitError("logical CKKS level requires a physical level below the "
                  "OperatorSpec lower_bound");
    return failure();
  }
  return static_cast<unsigned>(spec.upperBound -
                               consumedLogicalLevels * levelFactor);
}

FailureOr<Type> mapType(Type type, uint64_t logicalBase,
                        const PhysicalLevelSpec &spec, uint64_t levelFactor,
                        Operation *op) {
  auto tensor = type.dyn_cast<RankedTensorType>();
  if (!tensor)
    return type;
  auto poly = tensor.getElementType().dyn_cast<ckks::PolyTypeInterface>();
  if (!poly)
    return type;
  FailureOr<unsigned> level =
      mapLevel(poly.getLevel(), logicalBase, spec, levelFactor, op);
  if (failed(level))
    return failure();
  return RankedTensorType::get(tensor.getShape(), poly.switchLevel(*level),
                               tensor.getEncoding());
}

FailureOr<uint64_t> physicalLevel(Value value, uint64_t logicalBase,
                                  const PhysicalLevelSpec &spec,
                                  uint64_t levelFactor, Operation *op) {
  auto poly = ckks::getPolyType(value);
  if (!poly) {
    op->emitError("physical-level materialization requires CKKS poly values");
    return failure();
  }
  FailureOr<unsigned> level =
      mapLevel(poly.getLevel(), logicalBase, spec, levelFactor, op);
  if (failed(level))
    return failure();
  return static_cast<uint64_t>(*level);
}

struct MaterializePhysicalLevelsPass
    : public ckks::impl::MaterializePhysicalLevelsBase<
          MaterializePhysicalLevelsPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func->hasAttr("ckks.logical_init_level")) {
      func.emitError("CKKS physical levels have already been materialized");
      signalPassFailure();
      return;
    }
    auto logicalBaseAttr = func->getAttrOfType<IntegerAttr>("init_level");
    if (!logicalBaseAttr || logicalBaseAttr.getInt() < 0) {
      func.emitError("physical-level materialization requires a nonnegative "
                     "function init_level");
      signalPassFailure();
      return;
    }
    if (levelsPerLogicalLevel <= 0) {
      func.emitError("levels-per-logical-level must be positive");
      signalPassFailure();
      return;
    }
    const uint64_t logicalBase =
        static_cast<uint64_t>(logicalBaseAttr.getInt());
    const uint64_t levelFactor =
        static_cast<uint64_t>(levelsPerLogicalLevel.getValue());
    FailureOr<PhysicalLevelSpec> spec =
        readPhysicalLevelSpec(operatorSpecPath, func, levelFactor, bootProfile);
    if (failed(spec)) {
      signalPassFailure();
      return;
    }
    const auto plannedModuli = func->getAttrOfType<DenseI64ArrayAttr>("earth.native_scale_moduli");
    if (plannedModuli) {
      auto anchor = func->getAttrOfType<IntegerAttr>("earth.native_output_level");
      if (!spec->nativeRescale || plannedModuli.asArrayRef() != ArrayRef<int64_t>(spec->moduli)) {
        func.emitError("native scale-planning chain disagrees with physical-level OperatorSpec");
        return signalPassFailure();
      }
      if (!anchor || anchor.getInt() != static_cast<int64_t>(spec->upperBound)) {
        func.emitError("native scale-planning Boot anchor disagrees with physical-level OperatorSpec");
        return signalPassFailure();
      }
    }

    SmallVector<std::pair<Value, Type>> mappedValues;
    for (Block &block : func.getBody()) {
      for (BlockArgument argument : block.getArguments()) {
        FailureOr<Type> mapped =
            mapType(argument.getType(), logicalBase, *spec, levelFactor, func);
        if (failed(mapped)) {
          signalPassFailure();
          return;
        }
        mappedValues.emplace_back(argument, *mapped);
      }
      for (Operation &op : block) {
        for (Value result : op.getResults()) {
          FailureOr<Type> mapped =
              mapType(result.getType(), logicalBase, *spec, levelFactor, &op);
          if (failed(mapped)) {
            signalPassFailure();
            return;
          }
          if (isa<ckks::BootstrapCOp>(op) && spec->bootOutputScale) {
            auto tensor = cast<RankedTensorType>(*mapped);
            auto poly = cast<ckks::PolyType>(tensor.getElementType());
            if (poly.getLevel() != spec->upperBound ||
                poly.getScaleLog2() != *spec->bootOutputScale) {
              op.emitError("Boot result level/scale disagrees with selected profile; "
                           "set the matching bootstrap-scale and logical budget");
              return signalPassFailure();
            }
          }
          mappedValues.emplace_back(result, *mapped);
        }
      }
    }

    SmallVector<Type> functionResults;
    functionResults.reserve(func.getFunctionType().getNumResults());
    for (Type type : func.getFunctionType().getResults()) {
      FailureOr<Type> mapped =
          mapType(type, logicalBase, *spec, levelFactor, func);
      if (failed(mapped)) {
        signalPassFailure();
        return;
      }
      functionResults.push_back(*mapped);
    }

    SmallVector<std::pair<ckks::ModswitchCOp, int64_t>> modswitchFactors;
    WalkResult validation = func.walk([&](Operation *op) -> WalkResult {
      if (auto rescale = dyn_cast<ckks::RescaleCOp>(op)) {
        FailureOr<uint64_t> sourceLevel = physicalLevel(
            rescale.getSrc(), logicalBase, *spec, levelFactor, op);
        FailureOr<uint64_t> resultLevel = physicalLevel(
            rescale.getResult(), logicalBase, *spec, levelFactor, op);
        if (failed(sourceLevel) || failed(resultLevel))
          return WalkResult::interrupt();
        if (*sourceLevel <= *resultLevel ||
            *sourceLevel - *resultLevel > spec->maxRescaleLevels) {
          op->emitError("physical Rescale level drop is outside the target "
                        "OperatorSpec capability");
          return WalkResult::interrupt();
        }

        auto sourceType = ckks::getPolyType(rescale.getSrc());
        auto resultType = ckks::getPolyType(rescale.getResult());
        uint64_t modulusDrop = 0;
        for (uint64_t level = *resultLevel + 1; level <= *sourceLevel; ++level)
          modulusDrop += spec->modulusBits[level];
        if (sourceType.getScaleLog2() < resultType.getScaleLog2()) {
          op->emitError("logical Rescale scale drop must be nonnegative");
          return WalkResult::interrupt();
        }
        const uint64_t scaleDrop = static_cast<uint64_t>(
            sourceType.getScaleLog2() - resultType.getScaleLog2());
        // Exact power-of-two scale metadata requires a real integer
        // coefficient multiplication after native rescaling. The extra 30
        // nominal bits leave room for a large enough rounding factor; the
        // backend must still check actual primes and the resulting error.
        constexpr uint64_t integerAlignmentBits = 30;
        if (!plannedModuli && (spec->nativeRescale ? modulusDrop != scaleDrop :
            (modulusDrop < scaleDrop || modulusDrop - scaleDrop < integerAlignmentBits))) {
          op->emitError("physical Rescale requires at least 30 extra nominal "
                        "modulus bits for integer scale alignment");
          return WalkResult::interrupt();
        }
        uint64_t remainingModulusBits = 0;
        for (uint64_t level = 0; level <= *resultLevel; ++level)
          remainingModulusBits += spec->modulusBits[level];
        if (!plannedModuli && static_cast<uint64_t>(resultType.getScaleLog2()) >=
            remainingModulusBits) {
          op->emitError("physical Rescale target scale must be below the "
                        "remaining nominal modulus bits");
          return WalkResult::interrupt();
        }
      } else if (auto modswitch = dyn_cast<ckks::ModswitchCOp>(op)) {
        auto sourceType = ckks::getPolyType(modswitch.getSrc());
        auto resultType = ckks::getPolyType(modswitch.getResult());
        if (modswitch.getDownFactor() <= 0 ||
            sourceType.getLevel() <= resultType.getLevel() ||
            static_cast<uint64_t>(modswitch.getDownFactor()) !=
                sourceType.getLevel() - resultType.getLevel()) {
          op->emitError("logical ModSwitch downFactor does not match its level "
                        "drop");
          return WalkResult::interrupt();
        }
        if (static_cast<uint64_t>(modswitch.getDownFactor()) >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) /
                levelFactor) {
          op->emitError("physical ModSwitch downFactor overflows int64");
          return WalkResult::interrupt();
        }
        modswitchFactors.emplace_back(
            modswitch, static_cast<int64_t>(modswitch.getDownFactor()) *
                           static_cast<int64_t>(levelFactor));
      }
      return WalkResult::advance();
    });
    if (validation.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    for (auto &[value, type] : mappedValues)
      value.setType(type);
    SmallVector<Type> functionInputs;
    functionInputs.reserve(func.getNumArguments());
    for (BlockArgument argument : func.getArguments())
      functionInputs.push_back(argument.getType());
    func.setFunctionType(
        FunctionType::get(func.getContext(), functionInputs, functionResults));
    for (auto &[modswitch, downFactor] : modswitchFactors)
      modswitch.setDownFactor(downFactor);

    if (spec->nativeRescale && failed(materializeActualScales(func, *spec)))
      return signalPassFailure();
    Builder builder(func.getContext());
    func->setAttr("ckks.logical_init_level",
                  builder.getI64IntegerAttr(static_cast<int64_t>(logicalBase)));
    func->setAttr("ckks.levels_per_logical_level",
                  builder.getI64IntegerAttr(static_cast<int64_t>(levelFactor)));
    func->setAttr("init_level", builder.getI64IntegerAttr(
                                    static_cast<int64_t>(spec->upperBound)));
  }
};

} // namespace
