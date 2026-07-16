#include "hecate/Dialect/CKKS/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace hecate {
namespace ckks {
#define GEN_PASS_DEF_EMITHEVM
#include "hecate/Dialect/CKKS/Transforms/Passes.h.inc"
} // namespace ckks
} // namespace hecate

namespace {
struct EmitHEVMPass : public hecate::ckks::impl::EmitHEVMBase<EmitHEVMPass> {
  using Base::Base;

  void runOnOperation() override {
    getOperation().emitError(
        "HEVM emission has been retired; emit RuntimePlan JSON instead");
    signalPassFailure();
  }
};
} // namespace
