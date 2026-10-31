

#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "nlohmann/json.hpp"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include <fstream>
#include <cstdlib>

using namespace mlir;

namespace {
llvm::cl::opt<std::string> scalePolicy(
    "native-scale-policy", llvm::cl::desc("Native scale policy: threshold or regional"),
    llvm::cl::init("threshold"));
llvm::cl::opt<int64_t> nativeInputLevel(
    "native-input-level",
    llvm::cl::desc("Initial ciphertext level; defaults to the Boot output level"),
    llvm::cl::init(-1));
}

bool hecate::earth::usesRegionalScalePlanning() {
  if (scalePolicy != "threshold" && scalePolicy != "regional")
    llvm::report_fatal_error("native-scale-policy must be threshold or regional");
  return scalePolicy == "regional";
}

#define GET_TYPEDEF_CLASSES
#include "hecate/Dialect/Earth/IR/EarthOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "hecate/Dialect/Earth/IR/EarthOps.cpp.inc"

#include "hecate/Dialect/Earth/IR/EarthOpsDialect.cpp.inc"

#include "hecate/Dialect/Earth/IR/EarthCanonicalizerPattern.inc"

struct ScaleTypeTensorModel
    : public hecate::earth::HEScaleTypeInterface::ExternalModel<
          ScaleTypeTensorModel, mlir::RankedTensorType> {
  bool isCipher(Type t) const {
    if (auto scaleType = t.dyn_cast<mlir::RankedTensorType>()
                             .getElementType()
                             .dyn_cast<hecate::earth::HEScaleTypeInterface>()) {
      return scaleType.isCipher();
    } else {
      return false;
    }
  }
  hecate::earth::HEScaleTypeInterface toCipher(Type t) const {
    auto tt = t.dyn_cast<RankedTensorType>();
    return dyn_cast<hecate::earth::HEScaleTypeInterface>(RankedTensorType::get(
        tt.getShape(), tt.getElementType()
                           .dyn_cast<hecate::earth::HEScaleTypeInterface>()
                           .toCipher()));
  }
  hecate::earth::HEScaleTypeInterface toPlain(Type t) const {
    auto tt = t.dyn_cast<RankedTensorType>();
    return dyn_cast<hecate::earth::HEScaleTypeInterface>(RankedTensorType::get(
        tt.getShape(), tt.getElementType()
                           .dyn_cast<hecate::earth::HEScaleTypeInterface>()
                           .toPlain()));
  }
  hecate::earth::HEScaleTypeInterface switchScale(Type t,
                                                  unsigned scale) const {
    auto tt = t.dyn_cast<RankedTensorType>();
    return dyn_cast<hecate::earth::HEScaleTypeInterface>(RankedTensorType::get(
        tt.getShape(), tt.getElementType()
                           .dyn_cast<hecate::earth::HEScaleTypeInterface>()
                           .switchScale(scale)));
  }
  hecate::earth::HEScaleTypeInterface switchLevel(Type t,
                                                  unsigned level) const {
    auto tt = t.dyn_cast<RankedTensorType>();
    return dyn_cast<hecate::earth::HEScaleTypeInterface>(RankedTensorType::get(
        tt.getShape(), tt.getElementType()
                           .dyn_cast<hecate::earth::HEScaleTypeInterface>()
                           .switchLevel(level)));
  }
  unsigned getScale(Type t) const {
    if (auto scaleType = t.dyn_cast<mlir::RankedTensorType>()
                             .getElementType()
                             .dyn_cast<hecate::earth::HEScaleTypeInterface>()) {
      return scaleType.getScale();
    } else {
      return 0;
    }
  }
  unsigned getLevel(Type t) const {
    if (auto scaleType = t.dyn_cast<mlir::RankedTensorType>()
                             .getElementType()
                             .dyn_cast<hecate::earth::HEScaleTypeInterface>()) {
      return scaleType.getLevel();
    } else {
      return 0;
    }
  }
};

void hecate::earth::EarthDialect::initialize() {
  // Registers all the Types into the EVADialect class
  addTypes<
#define GET_TYPEDEF_LIST
#include "hecate/Dialect/Earth/IR/EarthOpsTypes.cpp.inc"
      >();

  // Registers all the Operations into the EVADialect class
  addOperations<
#define GET_OP_LIST
#include "hecate/Dialect/Earth/IR/EarthOps.cpp.inc"
      >();
  mlir::RankedTensorType::attachInterface<ScaleTypeTensorModel>(*getContext());
}

std::map<std::string, llvm::SmallVector<int64_t>> latencyTable;

std::map<std::string, llvm::SmallVector<double>> noiseTable;

int64_t hecate::earth::EarthDialect::polynomialDegree;
int64_t hecate::earth::EarthDialect::rescalingFactor;
int64_t hecate::earth::EarthDialect::bootstrapLevelLowerBound;
int64_t hecate::earth::EarthDialect::bootstrapLevelUpperBound;
int64_t hecate::earth::EarthDialect::levelUpperBound;
int64_t hecate::earth::EarthDialect::levelLowerBound;
std::vector<int64_t> hecate::earth::EarthDialect::nativeModuli;
hecate::ModulusChain hecate::earth::EarthDialect::nativeScaleChain;
int64_t hecate::earth::EarthDialect::nativeOutputLevel = -1;
int64_t hecate::earth::EarthDialect::nativeLevelAnchor = -1;
int64_t hecate::earth::EarthDialect::nativeLowerLevel = -1;
int64_t hecate::earth::EarthDialect::nativeBootInputLevel = -1;
double hecate::earth::EarthDialect::nativeBootInputScale = 0;

void hecate::earth::EarthDialect::setNativeScaleParameters(
    llvm::StringRef specPath, llvm::StringRef bootProfile) {
  auto fail = [](llvm::StringRef message) {
    llvm::errs() << "native scale planning: " << message << '\n';
    std::exit(1);
  };
  nativeModuli.clear();
  nativeScaleChain = {};
  nativeOutputLevel = nativeLowerLevel = nativeLevelAnchor = -1;
  nativeBootInputLevel = -1;
  nativeBootInputScale = 0;
  if (nativeInputLevel < -1) fail("native input level must be nonnegative");
  if (specPath.empty()) {
    if (nativeInputLevel >= 0) fail("native input level requires an actual native Q chain");
    return;
  }
  std::ifstream input(specPath.str());
  auto spec = nlohmann::json::parse(input);
  if (spec.at("rescale_mode") != "native") {
    if (nativeInputLevel >= 0) fail("native input level requires an actual native Q chain");
    return;
  }
  nativeModuli = spec.at("context").at("rns_moduli").get<std::vector<int64_t>>();
  nativeLowerLevel = spec.at("levels").at("lower_bound").get<int64_t>();
  for (const auto &profile : spec.at("boot_profiles"))
    if (profile.at("profile_id") == bootProfile.str()) {
      if (nativeOutputLevel >= 0)
        fail("duplicate Boot profile");
      nativeOutputLevel = profile.at("output_level").get<int64_t>();
      if (profile.contains("input_scale_log2")) {
        if (!profile.at("input_scale_log2").is_number_integer() ||
            profile.at("input_scale_log2").get<int64_t>() <= 0)
          fail("Boot input scale must be a positive integer");
        nativeBootInputLevel = profile.at("input_level_min").get<int64_t>();
        nativeBootInputScale = profile.at("input_scale_log2").get<double>();
        // Alignment is checked against each value's actual scale, rather
        // than reserving two Q for every arithmetic segment.
        nativeLowerLevel = std::max(nativeLowerLevel,
                                   nativeBootInputLevel);
      }
    }
  if (nativeOutputLevel < 0) fail("Boot profile is missing or invalid");
  nativeLevelAnchor = nativeInputLevel < 0 ? nativeOutputLevel : nativeInputLevel.getValue();
  if (nativeLevelAnchor < nativeOutputLevel ||
      nativeLevelAnchor >= static_cast<int64_t>(nativeModuli.size()))
    fail("native input level must cover the Boot output and fit the Q chain");
  if (nativeLowerLevel < 0 || nativeOutputLevel < nativeLowerLevel ||
      nativeOutputLevel >= static_cast<int64_t>(nativeModuli.size()))
    fail("invalid chain or Boot profile");
  for (auto q : nativeModuli)
    if (q <= 1 || q >= (int64_t{1} << 53))
      fail("moduli must be in (1, 2^53)");
  nativeScaleChain.reset(nativeModuli);
}

double hecate::earth::actualScale(Value value) {
  auto type = getScaleType(value);
  if (EarthDialect::nativeModuli.empty()) return type.getScale();
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    auto func = cast<func::FuncOp>(arg.getOwner()->getParentOp());
    if (auto scales = func->getAttrOfType<DenseF64ArrayAttr>("earth.arg_scales"))
      return scales.asArrayRef()[arg.getArgNumber()];
    return type.getScale();
  }
  auto *op = value.getDefiningOp();
  if (isa<ConstantOp>(op)) {
    if (auto encoding = op->getAttrOfType<FloatAttr>("earth.encoding_scale"))
      return encoding.getValueAsDouble();
    return type.getScale();
  }
  if (auto cached = op->getAttrOfType<FloatAttr>("earth.actual_scale"))
    return cached.getValueAsDouble();
  double scale;
  if (auto boot = dyn_cast<BootstrapOp>(op)) {
    auto planned = op->getAttrOfType<IntegerAttr>("planned_scale");
    scale = planned ? planned.getInt() : boot.getTargetScale();
  } else {
    scale = actualScale(op->getOperand(0));
    if (isa<MulOp>(op)) scale += actualScale(op->getOperand(1));
    else if (auto add = dyn_cast<AddOp>(op)) {
      // AddCP is encoded at the ciphertext scale. Native AddCC aligns the
      // lower actual scale to the higher one during physical lowering.
      if (getScaleType(add.getLhs()).isCipher() && getScaleType(add.getRhs()).isCipher())
        scale = std::max(scale, actualScale(add.getRhs()));
      else if (!getScaleType(add.getLhs()).isCipher())
        scale = actualScale(add.getRhs());
    } else if (auto rescale = dyn_cast<RescaleOp>(op)) {
      auto consumed = getScaleType(rescale.getValue()).getLevel();
      scale = EarthDialect::nativeScaleChain.rescale(
          scale, EarthDialect::nativeLevelAnchor - consumed);
    } else if (auto upscale = dyn_cast<UpscaleOp>(op)) {
      scale += upscale.getUpFactor();
    }
  }
  op->setAttr("earth.actual_scale", FloatAttr::get(Float64Type::get(op->getContext()), scale));
  return scale;
}

template <typename... Args>
void addOperationNamesTo(SmallVectorImpl<StringRef> &names) {
  (void)std::initializer_list<int>{
      0, (names.push_back(Args::getOperationName()), 0)...};
}

const std::map<std::string, llvm::SmallVector<int64_t>> &
hecate::earth::EarthDialect::getLatencyTable() {
  return latencyTable;
}
std::map<std::string, llvm::SmallVector<double>>
hecate::earth::EarthDialect::getNoiseTable() {
  return noiseTable;
}

void hecate::earth::EarthDialect::setCKKSParameters(llvm::StringRef filename) {
  // it should be read from file but currently fixed
  hecate::earth::EarthDialect::levelLowerBound = 1;  // 0 should be forbidden
  hecate::earth::EarthDialect::levelUpperBound = 13; // L =14
                                                     // inclusion
  hecate::earth::EarthDialect::polynomialDegree = 1LL << 15;

  SmallVector<StringRef, 4> names;
  addOperationNamesTo<
#define GET_OP_LIST
#include "hecate/Dialect/Earth/IR/EarthOps.cpp.inc"
      >(names);

  std::ifstream iff(filename.str());
  nlohmann::json config = nlohmann::json::parse(iff);
  hecate::earth::EarthDialect::rescalingFactor = config["rescalingFactor"];
  hecate::earth::EarthDialect::levelLowerBound = config["levelLowerBound"];
  hecate::earth::EarthDialect::levelUpperBound = config["levelUpperBound"];
  hecate::earth::EarthDialect::bootstrapLevelLowerBound =
      config["bootstrapLevelLowerBound"];
  hecate::earth::EarthDialect::bootstrapLevelUpperBound =
      config["bootstrapLevelUpperBound"];
  hecate::earth::EarthDialect::polynomialDegree = config["polynomialDegree"];

  for (auto &&name : names) {
    for (auto &&suffix : {"_single", "_double"}) {
      auto &latTab = latencyTable[name.str() + suffix];
      auto &noTab = noiseTable[name.str() + suffix];
      latTab.push_back(0);
      noTab.push_back(0);
      if (config["latencyTable"].contains(name.str() + suffix) &&
          config["latencyTable"][name.str() + suffix].size()) {
        for (auto &&data : config["latencyTable"][name.str() + suffix]) {
          latTab.push_back(data.get<int64_t>());
        }
      }
      latTab.resize(levelUpperBound + 1, latTab.back());

      if (config["noiseTable"].contains(name.str() + suffix) &&
          config["noiseTable"][name.str() + suffix].size()) {
        for (auto &&data : config["noiseTable"][name.str() + suffix]) {
          noTab.push_back(data.get<double>());
        }
      }
      noTab.resize(levelUpperBound + 1, noTab.back());

      latencyTable[name.str() + suffix] = latTab;
      noiseTable[name.str() + suffix] = noTab;
    }
  }
}

::mlir::LogicalResult hecate::earth::ConstantOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = ConstantOpAdaptor(operands, attributes, properties, regions);

  inferredReturnTypes.push_back(mlir::RankedTensorType::get(
      llvm::SmallVector<int64_t, 1>{1}, PlainType::get(context, 0, 0)));
  return ::mlir::success();
}

void hecate::earth::RescaleOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  // Replacing division by an actual q with subtraction of rescalingFactor
  // changes the physical scale, even for equal-width native primes.
  if (EarthDialect::nativeModuli.empty())
    patterns.add<RescaleUpscalePattern>(context);
}
void hecate::earth::RotateOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<ZeroRotatePattern>(context);
}

::mlir::LogicalResult hecate::earth::RescaleOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = RescaleOpAdaptor(operands, attributes, properties, regions);
  auto lScale = earth::getScaleType(op.getValue());
  if (!EarthDialect::nativeModuli.empty()) {
    auto index = EarthDialect::nativeLevelAnchor - lScale.getLevel();
    if (index < 1) return failure();
    double scale = EarthDialect::nativeScaleChain.rescale(actualScale(op.getValue()), index);
    if (scale < 0) return failure();
    inferredReturnTypes.push_back(lScale.switchLevel(lScale.getLevel() + 1)
                                      .switchScale(static_cast<unsigned>(std::ceil(scale))));
    return success();
  }
  inferredReturnTypes.push_back(
      lScale.switchLevel(lScale.getLevel() + 1)
          .switchScale(lScale.getScale() -
                       hecate::earth::EarthDialect::rescalingFactor));

  return ::mlir::success();
}

void hecate::earth::ModswitchOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<ModswitchModswitchPattern, ModswitchConstantPattern,
               ZeroModswitchPattern>(context);
}

::mlir::LogicalResult hecate::earth::ModswitchOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = ModswitchOpAdaptor(operands, attributes, properties, regions);
  auto lScale = earth::getScaleType(op.getValue());
  if (op.getDownFactor() >= 0) {
    inferredReturnTypes.push_back(
        lScale.switchLevel(lScale.getLevel() + op.getDownFactor()));
    return ::mlir::success();
  } else
    return ::mlir::failure();
}

void hecate::earth::UpscaleOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns
      .add<UpscaleUpscalePattern, UpscaleConstantPattern,
           UpscaleMulConstantPattern, ZeroUpscalePattern>(
          context);
}

::mlir::LogicalResult hecate::earth::UpscaleOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = UpscaleOpAdaptor(operands, attributes, properties, regions);
  auto lScale = earth::getScaleType(op.getValue());
  if (op.getUpFactor() >= 0) {
    inferredReturnTypes.push_back(
        lScale.switchScale(lScale.getScale() + op.getUpFactor()));
    return ::mlir::success();
  } else
    return ::mlir::failure();
}

void hecate::earth::BootstrapOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  /* patterns.add<exceedBootstrapBound>(context); */
}

::mlir::LogicalResult hecate::earth::BootstrapOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = BootstrapOpAdaptor(operands, attributes, properties, regions);
  auto lScale = earth::getScaleType(op.getValue());
  // accumulated scale > maximum scale limit
  const auto budget = EarthDialect::nativeModuli.empty()
      ? EarthDialect::bootstrapLevelUpperBound - EarthDialect::bootstrapLevelLowerBound
      : EarthDialect::nativeLevelAnchor - EarthDialect::nativeLowerLevel;
  if (lScale.getLevel() <= budget) {
    auto result = lScale.switchLevel(op.getTargetLevel());
    if (op.getTargetScale() > 0)
      result = result.switchScale(op.getTargetScale());
    inferredReturnTypes.push_back(result);
    return ::mlir::success();
  } else
    return ::mlir::failure();
}

void hecate::earth::AddOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<AddZeroPattern>(context);
}

::mlir::LogicalResult hecate::earth::AddOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = AddOpAdaptor(operands, attributes, properties, regions);
  auto lScale = earth::getScaleType(op.getLhs());
  auto lTensor = earth::getTensorType(op.getLhs());
  auto rScale = earth::getScaleType(op.getRhs());
  auto rTensor = earth::getTensorType(op.getRhs());
  if (usesRegionalScalePlanning() && lScale.isCipher() && rScale.isCipher() &&
      lScale.getLevel() == rScale.getLevel() && lTensor.getShape() == rTensor.getShape() &&
      std::abs(actualScale(op.getLhs()) - actualScale(op.getRhs())) <= 1e-8) {
    inferredReturnTypes.push_back(lScale.switchScale(std::max(lScale.getScale(), rScale.getScale())));
    return success();
  }
  if (lScale.getLevel() == rScale.getLevel() &&
      lScale.getScale() == rScale.getScale() &&
      lTensor.getShape()[0] == rTensor.getShape()[0]) {
    inferredReturnTypes.push_back(lScale.toCipher());
    return ::mlir::success();
  } else {
    return ::mlir::failure();
  }
}

void hecate::earth::MulOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  if (usesRegionalScalePlanning()) return;
  /* patterns.add<MulZeroPattern, MulOnePattern, NegMulPattern>(context); */
  patterns.add<MulOnePattern, NegMulPattern>(context);
}

::mlir::LogicalResult hecate::earth::MulOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = MulOpAdaptor(operands, attributes, properties, regions);
  auto lScale = earth::getScaleType(op.getLhs());
  auto lTensor = earth::getTensorType(op.getLhs());
  auto rScale = earth::getScaleType(op.getRhs());
  auto rTensor = earth::getTensorType(op.getRhs());
  if (lScale.getLevel() == rScale.getLevel() &&
      lTensor.getShape()[0] == rTensor.getShape()[0]) {
    inferredReturnTypes.push_back(
        lScale.switchScale(lScale.getScale() + rScale.getScale()).toCipher());
    return ::mlir::success();
  } else {
    return ::mlir::failure();
  }
}

mlir::RankedTensorType hecate::earth::getTensorType(mlir::Value v) {
  return v.getType().dyn_cast<mlir::RankedTensorType>();
}
hecate::earth::HEScaleTypeInterface hecate::earth::getScaleType(mlir::Value v) {
  return v.getType().dyn_cast<hecate::earth::HEScaleTypeInterface>();
}
