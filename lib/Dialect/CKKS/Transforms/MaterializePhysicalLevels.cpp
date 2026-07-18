#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "hecate/Dialect/CKKS/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "nlohmann/json.hpp"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <fstream>
#include <limits>
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
};

FailureOr<PhysicalLevelSpec> readPhysicalLevelSpec(llvm::StringRef path,
                                                   func::FuncOp func,
                                                   uint64_t levelFactor) {
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
      mode->get<std::string>() != "lazy") {
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
  result.lowerBound = static_cast<uint64_t>(signedLower);
  result.upperBound = static_cast<uint64_t>(signedUpper);
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
        readPhysicalLevelSpec(operatorSpecPath, func, levelFactor);
    if (failed(spec)) {
      signalPassFailure();
      return;
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
        if (sourceType.getScaleLog2() < resultType.getScaleLog2() ||
            static_cast<uint64_t>(sourceType.getScaleLog2() -
                                  resultType.getScaleLog2()) != modulusDrop) {
          op->emitError("logical Rescale scale drop does not match the target "
                        "physical modulus bits");
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
