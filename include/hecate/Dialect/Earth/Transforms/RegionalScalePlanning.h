#ifndef HECATE_REGIONAL_SCALE_PLANNING_H
#define HECATE_REGIONAL_SCALE_PLANNING_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <memory>

namespace hecate::earth {
bool usesRegionalScalePlanning();
void preserveRegionalOutputs(mlir::func::FuncOp func);
struct RegionalScaleState {
  mlir::Value value;
  unsigned q;
  double scale;
};
// Reuse one immutable region graph across coverage queries. No suffix cloning
// or re-partitioning is needed at each cut in a straight-line function.
class RegionalScaleAnalysis {
public:
  RegionalScaleAnalysis(mlir::func::FuncOp func, int64_t waterline,
                        int64_t bootScale);
  ~RegionalScaleAnalysis();
  mlir::FailureOr<int64_t> coverage(int64_t from,
      llvm::ArrayRef<RegionalScaleState> inputs, unsigned minimumQ = 0);
private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
mlir::LogicalResult scheduleRegionalRegions(mlir::func::FuncOp func);
// Public arithmetic boundaries; cuts inside a fused region would destroy its
// shared coefficient/Rescale plan and are not useful Boot search locations.
mlir::FailureOr<llvm::SmallVector<int64_t>>
regionalScaleBoundaries(mlir::func::FuncOp func);
// Clone only the requested slice, stopping at its live frontier. Original
// argument positions and operation IDs are preserved for the DaCapo DP.
mlir::func::FuncOp cloneRegionalSegment(mlir::func::FuncOp func,
                                        mlir::ValueRange inputs,
                                        mlir::ValueRange outputs,
                                        llvm::ArrayRef<int64_t> bootTargets,
                                        int64_t from, int64_t bootScale,
                                        bool intermediate);
// Scale search is bounded inside private arithmetic regions. The optional
// greedy scheduler refreshes a previously verified live frontier; it does not
// enumerate Boot choices as part of scale search.
mlir::LogicalResult planRegionalScales(mlir::func::FuncOp func,
                                       int64_t waterline, int64_t bootScale,
                                       bool insertBoots);
// Analyze a supplied segment without inserting Boot or rewriting its IR.
// Return the first regional boundary that cannot execute, or whose output
// falls below minimumQ; -1 means the complete segment was covered.
mlir::FailureOr<int64_t> analyzeRegionalScaleCoverage(mlir::func::FuncOp func,
                                                      int64_t waterline,
                                                      int64_t bootScale,
                                                      unsigned minimumQ = 0);
mlir::LogicalResult writeRegionalScaleReport(mlir::func::FuncOp func);
} // namespace hecate::earth
#endif
