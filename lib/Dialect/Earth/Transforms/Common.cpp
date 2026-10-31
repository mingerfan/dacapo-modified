#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Support/Support.h"
#include "hecate/Support/BootstrapInputAlignment.h"
#include "mlir/IR/Value.h"
#include <cmath>

using namespace mlir;

int64_t hecate::earth::bootstrapQBudget() {
  if (usesNativeScalePlanning())
    return EarthDialect::nativeLevelAnchor - EarthDialect::nativeLowerLevel;
  return EarthDialect::bootstrapLevelUpperBound -
         EarthDialect::bootstrapLevelLowerBound;
}

int64_t hecate::earth::bootstrapOutputLevel() {
  return usesNativeScalePlanning()
             ? EarthDialect::nativeLevelAnchor - EarthDialect::nativeOutputLevel
             : 0;
}

bool hecate::earth::usesNativeScalePlanning() {
  return !EarthDialect::nativeModuli.empty();
}

bool hecate::earth::canBootstrap(ScaleState state) {
  if (state.level < 0 || state.level > bootstrapQBudget())
    return false;
  if (!usesNativeScalePlanning() || EarthDialect::nativeBootInputScale == 0)
    return true;
  auto alignment = hecate::planBootstrapInputAlignment(
      EarthDialect::nativeScaleChain, EarthDialect::nativeLevelAnchor - state.level,
      state.scale, EarthDialect::nativeBootInputLevel,
      EarthDialect::nativeBootInputScale);
  return alignment.has_value();
}

hecate::earth::ScaleState hecate::earth::normalizeScaleState(
    ScaleState state, double waterline) {
  if (!usesNativeScalePlanning()) {
    while (state.scale >= waterline + EarthDialect::rescalingFactor) {
      state.scale -= EarthDialect::rescalingFactor;
      ++state.level;
    }
    return state;
  }
  if (state.level > EarthDialect::nativeLevelAnchor)
    return state;
  auto normalized = EarthDialect::nativeScaleChain.normalize(
      state.scale, EarthDialect::nativeLevelAnchor - state.level, waterline);
  state.scale = normalized.scale;
  state.level += normalized.drops;
  return state;
}

void hecate::earth::clearActualScales(func::FuncOp func) {
  func.walk([](Operation *op) { op->removeAttr("earth.actual_scale"); });
}

void hecate::earth::prepareThresholdOperands(ForwardMgmtInterface op,
                                             int64_t waterline) {
  op->removeAttr("earth.actual_scale");
  OpBuilder builder(op);
  for (auto &operand : op->getOpOperands())
    if (auto constant = operand.get().getDefiningOp<ConstantOp>();
        constant && !constant->getResult(0).hasOneUse())
      operand.set(builder.clone(*constant)->getResult(0));
  op.processOperandsEVA(waterline);
}

Value hecate::earth::normalizeScale(Value value, OpBuilder &builder,
                                    int64_t waterline, bool replaceUses) {
  IRRewriter rewriter(builder);
  auto state = normalizeScaleState(
      {static_cast<int64_t>(getScaleType(value).getLevel()), actualScale(value)}, waterline);
  // A trial can cross the working budget so coverage analysis can identify
  // its first infeasible boundary, but cannot consume the terminal q0.
  if (usesNativeScalePlanning() && state.level > EarthDialect::nativeLevelAnchor)
    return {};
  while (getScaleType(value).getLevel() < state.level) {
    auto next = builder.create<RescaleOp>(value.getLoc(), value);
    if (replaceUses)
      rewriter.replaceAllUsesExcept(value, next, next);
    value = next;
  }
  return value;
}

FailureOr<Value> hecate::earth::applyThresholdScale(ForwardMgmtInterface op,
                                                    int64_t waterline,
                                                    int64_t bootstrapScale,
                                                    bool checkBootInput) {
  if (waterline <= 0 || EarthDialect::rescalingFactor <= 0)
    return failure();
  OpBuilder builder(op);
  if (auto boot = dyn_cast<BootstrapOp>(op.getOperation())) {
    if (checkBootInput && !canBootstrap(
            {static_cast<int64_t>(getScaleType(boot.getValue()).getLevel()),
             actualScale(boot.getValue())})) {
      op->emitError("Boot input has insufficient Q/encoding precision for scale alignment");
      return failure();
    }
    if (auto planned = boot->getAttrOfType<IntegerAttr>("planned_scale");
        planned && planned.getInt() != bootstrapScale) {
      op->emitError("Boot output scale disagrees with scale-management policy");
      return failure();
    }
    if (!boot->hasAttr("planned_scale") && boot.getTargetScale() > 0 &&
        boot.getTargetScale() != bootstrapScale) {
      op->emitError("Boot output scale disagrees with scale-management policy");
      return failure();
    }
    boot.setTargetLevel(bootstrapOutputLevel());
    boot.setTargetScale(bootstrapScale);
  } else {
    prepareThresholdOperands(op, waterline);
  }
  SmallVector<Type> types;
  op->removeAttr("earth.actual_scale");
  auto inference = cast<InferTypeOpInterface>(op.getOperation());
  if (failed(inference.inferReturnTypes(
          op->getContext(), op->getLoc(), op->getOperands(),
          op->getAttrDictionary(), op->getPropertiesStorage(), op->getRegions(),
          types)))
    return failure();
  op->getResult(0).setType(types.front());
  builder.setInsertionPointAfter(op);
  Value result = op->getResult(0);
  if (isa<MulOp, BootstrapOp>(op.getOperation()))
    result = normalizeScale(result, builder, waterline, true);
  if (!result) return failure();
  return result;
}

void hecate::earth::finishThresholdScale(func::FuncOp func, OpBuilder &builder,
                                         TypeRange inputTypes) {
  auto result = cast<func::ReturnOp>(func.front().getTerminator());
  if (usesNativeScalePlanning()) {
    clearActualScales(func);
    SmallVector<double> scales;
    for (Value value : result.getOperands()) scales.push_back(actualScale(value));
    func->setAttr("earth.result_scales", builder.getDenseF64ArrayAttr(scales));
    func->setAttr("earth.native_scale_moduli", builder.getDenseI64ArrayAttr(EarthDialect::nativeModuli));
    func->setAttr("earth.native_output_level", builder.getI64IntegerAttr(EarthDialect::nativeOutputLevel));
    func->setAttr("earth.native_level_anchor", builder.getI64IntegerAttr(EarthDialect::nativeLevelAnchor));
  }
  func.setFunctionType(
      builder.getFunctionType(inputTypes, result.getOperandTypes()));
  func->setAttr("init_level", builder.getI64IntegerAttr(
      usesNativeScalePlanning() ? EarthDialect::nativeLevelAnchor : EarthDialect::bootstrapLevelUpperBound));
  SmallVector<int64_t> inputScales, outputScales;
  for (Value arg : func.getArguments())
    inputScales.push_back(getScaleType(arg).getScale());
  for (Value value : result.getOperands())
    outputScales.push_back(getScaleType(value).getScale());
  func->setAttr("arg_scale", builder.getDenseI64ArrayAttr(inputScales));
  func->setAttr("res_scale", builder.getDenseI64ArrayAttr(outputScales));
}

void hecate::earth::refineLevel(mlir::OpBuilder builder, mlir::Operation *op,
                                int64_t waterline, int64_t output_val,
                                int64_t min_level) {
  int64_t max_required_level =
      hecate::earth::EarthDialect::bootstrapLevelUpperBound - min_level;
  if (max_required_level < 0) {
    max_required_level =
        hecate::earth::EarthDialect::levelUpperBound - min_level;
  }

  builder.setInsertionPoint(op);

  int64_t acc_scale_max = 0;
  int64_t rescalingFactor = hecate::earth::EarthDialect::rescalingFactor;
  for (auto v : op->getOperands()) {
    auto st = v.getType().dyn_cast<hecate::earth::HEScaleTypeInterface>();
    auto acc_scale = st.getLevel() * rescalingFactor + st.getScale();
    acc_scale_max = std::max(acc_scale_max, acc_scale);
  }
  for (size_t i = 0; i < op->getNumOperands(); i++) {
    auto v = op->getOperand(i);
    if (hecate::getIntegerAttr("is_bypassed", v) > 0) {
      /* llvm::errs() << "pass early modswitch \n"; */
      /* v.dump(); */
      continue;
    }
    auto st = v.getType().dyn_cast<hecate::earth::HEScaleTypeInterface>();
    auto acc_scale =
        st.getLevel() * rescalingFactor + st.getScale() + output_val;
    auto max_acc_scale = max_required_level * rescalingFactor;
    int64_t level_diff = (max_acc_scale - acc_scale) / rescalingFactor;
    op->setOperand(i, builder.create<hecate::earth::ModswitchOp>(
                          op->getLoc(), v, level_diff));
  }
}

void hecate::earth::refineReturnValues(mlir::func::FuncOp func,
                                       mlir::OpBuilder builder,
                                       SmallVector<mlir::Type, 4> inputTypes,
                                       int64_t waterline, int64_t output_val) {

  int64_t max_required_level =
      hecate::earth::EarthDialect::bootstrapLevelUpperBound;
  if (max_required_level < 0)
    max_required_level = hecate::earth::EarthDialect::levelUpperBound;

  // Reduce the level of the resulting values to reduce the size of returns
  //
  auto rop = dyn_cast<func::ReturnOp>(func.getBlocks().front().getTerminator());

  if (func->hasAttr("is_mid_segment") &&
      func->getAttrOfType<mlir::BoolAttr>("is_mid_segment").getValue()) {
    auto &&bypassTypes = func->getAttr("segment_returnBypasses")
                             .dyn_cast<mlir::ArrayAttr>()
                             .getValue();
    for (size_t i = 0; i < rop->getNumOperands(); i++) {
      auto v = rop->getOperand(i);
      bool isBypass = bypassTypes[i].dyn_cast<mlir::BoolAttr>().getValue();
      hecate::setIntegerAttr("is_bypassed", v, isBypass);
    }
    hecate::earth::refineLevel(
        builder, rop, waterline, 0,
        hecate::earth::EarthDialect::bootstrapLevelLowerBound - 1);

  } else
    hecate::earth::refineLevel(builder, rop, waterline, output_val, 0);

  func.walk([&](hecate::earth::BootstrapOp bop) {
    hecate::earth::refineLevel(
        builder, bop, waterline, 0,
        hecate::earth::EarthDialect::bootstrapLevelLowerBound - 1);
  });

  /* func.walk([&](func::ReturnOp rop) { */
  // Remap the return types
  func.setFunctionType(
      builder.getFunctionType(inputTypes, rop.getOperandTypes()));
  /* }); */

  func->setAttr("init_level", builder.getI64IntegerAttr(max_required_level));

  SmallVector<int64_t, 4> scales_in;
  SmallVector<int64_t, 4> scales_out;

  for (auto &&arg : func.getArguments()) {
    scales_in.push_back(arg.getType()
                            .dyn_cast<hecate::earth::HEScaleTypeInterface>()
                            .getScale());
  }
  func->setAttr("arg_scale", builder.getDenseI64ArrayAttr(scales_in));
  for (auto &&restype : func.getResultTypes()) {
    scales_out.push_back(
        restype.dyn_cast<hecate::earth::HEScaleTypeInterface>().getScale());
  }
  func->setAttr("res_scale", builder.getDenseI64ArrayAttr(scales_out));
}

/* llvm::SmallVector<mlir::Type, 4> */
void hecate::earth::refineInputValues(mlir::func::FuncOp func,
                                      mlir::OpBuilder builder,
                                      SmallVector<mlir::Type, 4> &inputTypes,
                                      int64_t waterline, int64_t output_val) {
  if (usesNativeScalePlanning()) clearActualScales(func);
  if (usesNativeScalePlanning())
    func->setAttr("earth.scale_waterline", builder.getF64FloatAttr(waterline));
  // Set function argument types
  if (!func->hasAttr("segment_inputType")) {
    if (usesNativeScalePlanning())
      func->setAttr("earth.arg_scales", builder.getDenseF64ArrayAttr(
          SmallVector<double>(func.getNumArguments(), static_cast<double>(waterline))));
    for (auto argval : func.getArguments()) {
      auto tt = argval.getType().dyn_cast<RankedTensorType>();
      argval.setType(RankedTensorType::get(
          tt.getShape(), tt.getElementType()
                             .dyn_cast<hecate::earth::HEScaleTypeInterface>()
                             .switchScale(waterline)));
      inputTypes.push_back(argval.getType());
    }
  } else {
    if (usesNativeScalePlanning())
      func->setAttr("earth.arg_scales", func->getAttr("segment_input_scales"));
    auto &&inputType_attrs = func->getAttr("segment_inputType")
                                 .dyn_cast<mlir::ArrayAttr>()
                                 .getValue();
    for (size_t i = 0; i < func.getNumArguments(); i++) {
      auto argval = func.getArgument(i);
      auto input_type = inputType_attrs[i]
                            .dyn_cast<mlir::TypeAttr>()
                            .getValue()
                            .dyn_cast<hecate::earth::HEScaleTypeInterface>();
      argval.setType(input_type);
      inputTypes.push_back(argval.getType());
    }
  }
  return;
}

void hecate::earth::inferTypeForward(hecate::earth::ForwardMgmtInterface sop) {
  Operation *oop = sop.getOperation();
  auto iop = dyn_cast<mlir::InferTypeOpInterface>(oop);
  SmallVector<Type, 4> retTypes;
  if (iop.inferReturnTypes(oop->getContext(), oop->getLoc(), oop->getOperands(),
                           oop->getAttrDictionary(),
                           oop->getPropertiesStorage(), oop->getRegions(),
                           retTypes)
          .succeeded()) {
    oop->getResults().back().setType(retTypes.back());
  }
}

llvm::SmallVector<mlir::Value, 4>
hecate::earth::attachOpid(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::Value, 4> values;
  // attach the opid to operation
  values.push_back(nullptr);
  for (Value argument : func.getArguments())
    values.push_back(argument);
  func->walk([&](hecate::earth::HEScaleOpInterface sop) {
    if ((llvm::isa<hecate::earth::UpscaleOp>(sop) ||
         llvm::isa<hecate::earth::RescaleOp>(sop) ||
         llvm::isa<hecate::earth::BootstrapOp>(sop) ||
         llvm::isa<hecate::earth::ModswitchOp>(sop))) {
      /* assert(0 && "Currently not supported"); */
      return;
    }
    for (auto &&val : sop.getOperation()->getResults()) {
      values.push_back(val);
    }
  });
  return values;
}

int64_t hecate::earth::calcWaterline(hecate::ScaleManagementUnit &smu,
                                     mlir::Value v, int64_t waterline) {
  while (smu.getID(v) < 0) {
    // Backward movement for -1 value because it means scale management
    // operations or apply_schedule op.
    if (v.getUses().empty() || !v.getUses().begin().getUser()->getNumResults())
      return waterline;
    v = v.getUses().begin().getUser()->getResult(0);
  }
  return smu.inNoisyGroup(v) ? waterline + 4 : waterline;
}

int64_t hecate::earth::calcWaterline(hecate::ScaleManagementUnit &smu,
                                     mlir::Operation *op, int64_t waterline) {
  return calcWaterline(smu, op->getResult(0), waterline);
}
