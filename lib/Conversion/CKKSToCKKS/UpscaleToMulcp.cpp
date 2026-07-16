

#include "hecate/Conversion/CKKSToCKKS/UpscaleToMulcp.h"
#include "hecate/Conversion/CKKSCommon/PolyTypeConverter.h"

#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "mlir/Conversion/ArithCommon/AttrToLLVMConverter.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <type_traits>

namespace hecate {
#define GEN_PASS_DEF_UPSCALETOMULCPCONVERSION
#include "hecate/Conversion/Passes.h.inc"
} // namespace hecate

using namespace mlir;
using namespace hecate;

namespace {

//===----------------------------------------------------------------------===//
// Straightforward Op Lowerings
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Op Lowering Patterns
//===----------------------------------------------------------------------===//

/// Directly lower to LLVM op.

struct UpscaleCOpLowering
    : public OpConversionPattern<hecate::ckks::UpscaleCOp> {
  using OpConversionPattern<hecate::ckks::UpscaleCOp>::ConversionPattern;
  UpscaleCOpLowering(MLIRContext *ctxt)
      : OpConversionPattern<hecate::ckks::UpscaleCOp>(ctxt) {}

  LogicalResult
  matchAndRewrite(hecate::ckks::UpscaleCOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace

//===----------------------------------------------------------------------===//
// UpscaleOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
UpscaleCOpLowering::matchAndRewrite(hecate::ckks::UpscaleCOp op,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  auto srcType = adaptor.getSrc()
                     .getType()
                     .cast<RankedTensorType>()
                     .getElementType()
                     .cast<hecate::ckks::PolyTypeInterface>();
  auto plainType = srcType.switchComponents(1).switchScaleLog2(
      static_cast<unsigned>(adaptor.getUpFactor()));

  auto dst = rewriter.create<tensor::EmptyOp>(
      op.getLoc(), op.getType().getShape(), plainType);

  auto payloadType =
      RankedTensorType::get({hecate::earth::EarthDialect::polynomialDegree / 2},
                            rewriter.getF64Type());
  auto payload =
      DenseElementsAttr::get(payloadType, rewriter.getF64FloatAttr(1.0));

  auto rhs = rewriter.create<ckks::EncodeOp>(op.getLoc(), dst, payload);

  rewriter.replaceOpWithNewOp<ckks::MulCPOp>(op, adaptor.getDst(),
                                             adaptor.getSrc(), rhs);
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct UpscaleToMulcpConversion
    : public hecate::impl::UpscaleToMulcpConversionBase<
          UpscaleToMulcpConversion> {

  using Base::Base;

  void runOnOperation() override {
    ConversionTarget target(getContext());

    mlir::RewritePatternSet patterns(&getContext());

    target.addIllegalOp<hecate::ckks::UpscaleCOp>();
    target.addLegalDialect<hecate::ckks::CKKSDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<func::FuncDialect>();

    hecate::ckks::populateUpscaleToMulcpConversionPatterns(&getContext(),
                                                           patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//
//
void hecate::ckks::populateUpscaleToMulcpConversionPatterns(
    mlir::MLIRContext *ctxt, mlir::RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<UpscaleCOpLowering> (ctxt);

  // clang-format on
}
std::unique_ptr<::mlir::OperationPass<::mlir::func::FuncOp>>
hecate::ckks::createUpscaleToMulcpConversionPass() {
  return std::make_unique<UpscaleToMulcpConversion>();
}
