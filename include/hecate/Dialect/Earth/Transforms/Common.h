
#ifndef HECATE_EARTH_TRANSFROMS_COMMON
#define HECATE_EARTH_TRANSFROMS_COMMON

#include "hecate/Dialect/Earth/Analysis/ScaleManagementUnit.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace hecate {
namespace earth {
void refineLevel(mlir::OpBuilder builder, mlir::Operation *op,
                 int64_t waterline, int64_t output_val, int64_t min_level);
void refineInputValues(mlir::func::FuncOp func, mlir::OpBuilder builder,
                       llvm::SmallVector<mlir::Type, 4> &inputTypes,
                       int64_t waterline, int64_t output_val);

void refineReturnValues(mlir::func::FuncOp func, mlir::OpBuilder builder,
                        llvm::SmallVector<mlir::Type, 4> inputTypes,
                        int64_t waterline, int64_t output_val);
void inferTypeForward(hecate::earth::ForwardMgmtInterface sop);

// Logical levels count consumed Q. Each threshold Rescale consumes one Q;
// Boot starts a new chain with its explicitly selected output scale.
int64_t bootstrapQBudget();
bool usesNativeScalePlanning();
struct ScaleState {
  int64_t level; // Q consumed since the selected Boot output anchor.
  double scale;  // Actual log2(scale), independent of the integer IR type.
};
double actualScale(mlir::Value value);
ScaleState normalizeScaleState(ScaleState state, double waterline);
void clearActualScales(mlir::func::FuncOp func);
void prepareThresholdOperands(ForwardMgmtInterface op, int64_t waterline);
mlir::Value normalizeScale(mlir::Value value, mlir::OpBuilder &builder,
                           int64_t waterline, bool replaceUses = false);
mlir::FailureOr<mlir::Value>
applyThresholdScale(hecate::earth::ForwardMgmtInterface op, int64_t waterline,
                    int64_t bootstrapScale);
void finishThresholdScale(mlir::func::FuncOp func, mlir::OpBuilder &builder,
                          mlir::TypeRange inputTypes);

llvm::SmallVector<mlir::Value, 4> attachOpid(mlir::func::FuncOp func);

// To Calculate the waterline for SNR Rescaling
int64_t calcWaterline(hecate::ScaleManagementUnit &smu, mlir::Value v,
                      int64_t waterline);
int64_t calcWaterline(hecate::ScaleManagementUnit &smu, mlir::Operation *op,
                      int64_t waterline);

} // namespace earth
} // namespace hecate

#endif
