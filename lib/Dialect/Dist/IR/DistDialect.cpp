#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "hecate/Dialect/Dist/IR/DistOps.h"

using namespace mlir;

#define GET_TYPEDEF_CLASSES
#include "hecate/Dialect/Dist/IR/DistOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "hecate/Dialect/Dist/IR/DistOps.cpp.inc"

#include "hecate/Dialect/Dist/IR/DistOpsDialect.cpp.inc"

void hecate::dist::DistDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hecate/Dialect/Dist/IR/DistOps.cpp.inc"
      >();
}

LogicalResult hecate::dist::TransferOp::verify() {
  if (!hecate::ckks::getPolyType(getInput()))
    return emitOpError("input must be a CKKS polynomial value");
  const int64_t transferId = getTransferIdAttr().getInt();
  const int64_t sourceRank = getSourceRankAttr().getInt();
  const int64_t sourceDevice = getSourceDeviceAttr().getInt();
  const int64_t destinationRank = getDestinationRankAttr().getInt();
  const int64_t destinationDevice = getDestinationDeviceAttr().getInt();
  if (transferId < 0 || sourceRank < 0 || destinationRank < 0)
    return emitOpError("transfer id and ranks must be nonnegative");
  if (sourceDevice < -1 || destinationDevice < -1)
    return emitOpError("device index must be -1 for Host or nonnegative");
  if (sourceRank == destinationRank && sourceDevice == destinationDevice)
    return emitOpError("source and destination must differ");
  return success();
}
