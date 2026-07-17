#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "hecate/Dialect/CKKS/Transforms/Passes.h"
#include "hecate/Dialect/Dist/IR/DistOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"

#include <cstdint>
#include <map>
#include <tuple>

namespace hecate {
namespace ckks {
#define GEN_PASS_DEF_MATERIALIZECOMMUNICATION
#include "hecate/Dialect/CKKS/Transforms/Passes.h.inc"
} // namespace ckks
} // namespace hecate

using namespace mlir;
namespace ckks = hecate::ckks;
namespace dist = hecate::dist;

namespace {

struct Place {
  int64_t rank = 0;
  int64_t device = -1;

  bool operator==(const Place &other) const {
    return rank == other.rank && device == other.device;
  }
  bool operator<(const Place &other) const {
    return std::tie(rank, device) < std::tie(other.rank, other.device);
  }
};

FailureOr<Place> placeOf(Value value, func::FuncOp func,
                         Operation *diagnostic) {
  if (auto argument = value.dyn_cast<BlockArgument>()) {
    auto rank = func.getArgAttrOfType<IntegerAttr>(argument.getArgNumber(),
                                                   "dist.rank");
    auto device = func.getArgAttrOfType<IntegerAttr>(argument.getArgNumber(),
                                                     "dist.device");
    if (!rank || !device) {
      diagnostic->emitError("function argument is missing placement");
      return failure();
    }
    return Place{rank.getInt(), device.getInt()};
  }
  Operation *definition = value.getDefiningOp();
  auto rank = definition->getAttrOfType<IntegerAttr>("dist.rank");
  auto device = definition->getAttrOfType<IntegerAttr>("dist.device");
  if (!rank || !device) {
    diagnostic->emitError("operand definition is missing placement");
    return failure();
  }
  return Place{rank.getInt(), device.getInt()};
}

bool isInitializationValue(Value value) {
  return value.isa<BlockArgument>() ||
         isa_and_nonnull<ckks::EncodeOp>(value.getDefiningOp());
}

dist::TransferOp createTransfer(OpBuilder &builder, Location location,
                                Value input, int64_t transferId,
                                const Place &source,
                                const Place &destination,
                                bool initialization) {
  OperationState state(location, dist::TransferOp::getOperationName());
  state.addOperands(input);
  state.addTypes(input.getType());
  state.addAttribute("transfer_id", builder.getI64IntegerAttr(transferId));
  state.addAttribute("source_rank", builder.getI64IntegerAttr(source.rank));
  state.addAttribute("source_device",
                     builder.getI64IntegerAttr(source.device));
  state.addAttribute("destination_rank",
                     builder.getI64IntegerAttr(destination.rank));
  state.addAttribute("destination_device",
                     builder.getI64IntegerAttr(destination.device));
  state.addAttribute("initialization", builder.getBoolAttr(initialization));
  Operation *created = builder.create(state);
  created->setAttr("dist.rank",
                   builder.getI64IntegerAttr(destination.rank));
  created->setAttr("dist.device",
                   builder.getI64IntegerAttr(destination.device));
  return cast<dist::TransferOp>(created);
}

struct MaterializeCommunicationPass
    : public hecate::ckks::impl::MaterializeCommunicationBase<
          MaterializeCommunicationPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (!func->hasAttr("dist.device_counts")) {
      func.emitError("communication materialization requires placement");
      signalPassFailure();
      return;
    }
    if (!func.getBody().hasOneBlock()) {
      func.emitError("communication materialization requires one block");
      signalPassFailure();
      return;
    }
    for (Operation &op : func.getBody().front()) {
      if (isa<dist::TransferOp>(op)) {
        op.emitError("communication has already been materialized");
        signalPassFailure();
        return;
      }
    }

    llvm::DenseMap<Value, std::map<Place, Value>> copies;
    int64_t nextTransferId = 0;
    Block &block = func.getBody().front();
    for (Operation &op : llvm::make_early_inc_range(block)) {
      if (op.getName().getDialectNamespace() != "ckks" ||
          isa<ckks::EncodeOp>(op))
        continue;
      auto rank = op.getAttrOfType<IntegerAttr>("dist.rank");
      auto device = op.getAttrOfType<IntegerAttr>("dist.device");
      if (!rank || !device) {
        op.emitError("compute operation is missing placement");
        signalPassFailure();
        return;
      }
      const Place destination{rank.getInt(), device.getInt()};
      for (OpOperand &operand : op.getOpOperands()) {
        Value input = operand.get();
        FailureOr<Place> source = placeOf(input, func, &op);
        if (failed(source)) {
          signalPassFailure();
          return;
        }
        if (*source == destination)
          continue;
        auto &copiesAtPlaces = copies[input];
        auto existing = copiesAtPlaces.find(destination);
        if (existing != copiesAtPlaces.end()) {
          operand.set(existing->second);
          continue;
        }
        OpBuilder builder(&op);
        dist::TransferOp transfer = createTransfer(
            builder, op.getLoc(), input, nextTransferId++, *source,
            destination, isInitializationValue(input));
        copiesAtPlaces.emplace(destination, transfer.getResult());
        operand.set(transfer.getResult());
      }
    }

    for (Operation &op : block) {
      if (op.getName().getDialectNamespace() != "ckks" ||
          isa<ckks::EncodeOp>(op))
        continue;
      const Place operationPlace{
          op.getAttrOfType<IntegerAttr>("dist.rank").getInt(),
          op.getAttrOfType<IntegerAttr>("dist.device").getInt()};
      for (Value operand : op.getOperands()) {
        FailureOr<Place> inputPlace = placeOf(operand, func, &op);
        if (failed(inputPlace) || !(*inputPlace == operationPlace)) {
          op.emitError("communication materialization left a remote operand");
          signalPassFailure();
          return;
        }
      }
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ckks::CKKSDialect, dist::DistDialect>();
  }
};

} // namespace
