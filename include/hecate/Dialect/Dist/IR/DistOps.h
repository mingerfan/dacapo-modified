#ifndef HECATE_DIALECT_DIST_IR_DISTOPS_H
#define HECATE_DIALECT_DIST_IR_DISTOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "hecate/Dialect/Dist/IR/DistOpsDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "hecate/Dialect/Dist/IR/DistOpsTypes.h.inc"
#define GET_OP_CLASSES
#include "hecate/Dialect/Dist/IR/DistOps.h.inc"

#endif
