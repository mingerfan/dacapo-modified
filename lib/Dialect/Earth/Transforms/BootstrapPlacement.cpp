#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "hecate/Support/Support.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace hecate::earth {
#define GEN_PASS_DEF_BOOTSTRAPPLACEMENT
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace hecate::earth
using namespace mlir;

namespace {
struct BootstrapPlacementPass
    : hecate::earth::impl::BootstrapPlacementBase<BootstrapPlacementPass> {
  BootstrapPlacementPass() = default;
  BootstrapPlacementPass(hecate::earth::BootstrapPlacementOptions options) {
    bootstrap_scale = options.bootstrap_scale;
  }
  void runOnOperation() override {
    auto func = getOperation();
    OpBuilder builder(func);
    auto values = hecate::earth::attachOpid(func);
    SmallVector<std::pair<int64_t, SmallVector<int64_t>>> schedule;
    if (auto entries = func->getAttrOfType<ArrayAttr>("btp_schedule")) {
      for (auto entry : entries) {
        auto data = cast<DenseI64ArrayAttr>(entry).asArrayRef();
        if (data.empty()) {
          func.emitError("empty Boot schedule entry");
          return signalPassFailure();
        }
        schedule.push_back(
            {data.front(), SmallVector<int64_t>(data.drop_front())});
      }
    } else {
      auto targets = func->getAttrOfType<DenseI64ArrayAttr>("btp_target");
      if (!targets) {
        func.emitError("missing Boot targets or schedule");
        return signalPassFailure();
      }
      int64_t cut = 0;
      if (auto edges = func->getAttrOfType<DenseI64ArrayAttr>("cutted_edge"))
        cut = edges.asArrayRef().front();
      if (auto at = func->getAttrOfType<IntegerAttr>("btp_at"))
        cut = at.getInt();
      schedule.push_back({cut, SmallVector<int64_t>(targets.asArrayRef())});
    }
    DenseMap<Value, Value> current;
    int64_t previousCut = -1;
    for (const auto &[cut, targets] : schedule) {
      if (cut < previousCut || cut < 0 ||
          cut >= static_cast<int64_t>(values.size()) ||
          (cut && !values[cut].getDefiningOp())) {
        func.emitError("invalid or unordered Boot boundary");
        return signalPassFailure();
      }
      previousCut = cut;
      Operation *boundary = cut ? values[cut].getDefiningOp() : nullptr;
      if (boundary)
        builder.setInsertionPointAfter(boundary);
      else
        builder.setInsertionPointToStart(&func.front());
      for (auto id : targets) {
        if (id <= 0 || id >= static_cast<int64_t>(values.size()) ||
            (cut && id > cut) || (!cut && id > func.getNumArguments()) ||
            !hecate::earth::getScaleType(values[id]).isCipher()) {
          func.emitError("invalid ciphertext target at Boot boundary");
          return signalPassFailure();
        }
        auto logical = values[id];
        Value source =
            current.count(logical) ? current.lookup(logical) : logical;
        // Preserve valid intermediate types until threshold management updates
        // the Boot and its consumers together. planned_scale is the contract.
        auto type = hecate::earth::getScaleType(source);
        auto boot = builder.create<hecate::earth::BootstrapOp>(
            source.getLoc(), source, type.getLevel(), type.getScale());
        boot->setAttr("planned_scale",
                      builder.getI64IntegerAttr(
                          bootstrap_scale
                              ? bootstrap_scale.getValue()
                              : hecate::earth::EarthDialect::rescalingFactor));
        boot->setAttr("boot_target", builder.getI64IntegerAttr(id));
        hecate::setIntegerAttr("opid", boot.getResult(), cut);
        source.replaceUsesWithIf(boot.getResult(), [&](OpOperand &use) {
          return use.getOwner() != boot.getOperation() &&
                 (!boundary || boundary->isBeforeInBlock(use.getOwner()));
        });
        current[logical] = boot;
      }
    }
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
