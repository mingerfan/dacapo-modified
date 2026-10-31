#ifndef HECATE_REGIONAL_SCALE_PLANNING_H
#define HECATE_REGIONAL_SCALE_PLANNING_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace hecate::earth {
bool usesRegionalScalePlanning();
// Scale search is bounded inside private arithmetic regions. The optional
// greedy scheduler refreshes a previously verified live frontier; it does not
// enumerate Boot choices as part of scale search.
mlir::LogicalResult planRegionalScales(mlir::func::FuncOp func,
                                       int64_t waterline, int64_t bootScale,
                                       bool insertBoots);
} // namespace hecate::earth
#endif
