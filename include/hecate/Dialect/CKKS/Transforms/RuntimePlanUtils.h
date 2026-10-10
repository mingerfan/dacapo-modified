#ifndef HECATE_RUNTIME_PLAN_UTILS_H
#define HECATE_RUNTIME_PLAN_UTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include <cstdint>
#include <map>
#include <vector>

namespace hecate::ckks {

struct RuntimePlace {
  int64_t rank = 0;
  int64_t device = -1;
  bool operator==(const RuntimePlace &other) const {
    return rank == other.rank && device == other.device;
  }
  bool operator<(const RuntimePlace &other) const {
    return rank != other.rank ? rank < other.rank : device < other.device;
  }
};

struct RuntimeInstruction {
  mlir::Operation *operation;
  bool initialization;
  uint64_t ordinal;
};

struct RuntimePlanLayout {
  llvm::DenseMap<mlir::Value, uint64_t> ids;
  std::map<uint64_t, mlir::Value> values;
  std::vector<RuntimeInstruction> instructions;
};

bool isRuntimeInitialization(mlir::Operation *operation);
mlir::FailureOr<RuntimePlace> runtimePlace(mlir::Value value,
                                           mlir::func::FuncOp func);
mlir::FailureOr<RuntimePlanLayout>
buildRuntimePlanLayout(mlir::func::FuncOp func);
bool canReuseRuntimeInput(mlir::Operation *operation, mlir::func::FuncOp func);

} // namespace hecate::ckks
#endif
