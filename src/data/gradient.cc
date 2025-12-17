/**
 * Copyright 2025, XGBoost Contributors
 */
#include "xgboost/gradient.h"

#include "../common/threading_utils.h"  // for ParallelFor
#include "array_interface.h"            // for ArrayInterface
#include "array_page_source.h"
#include "xgboost/context.h"  // for Context
#include "xgboost/linalg.h"   // for Matrix

#if !defined(XGBOOST_USE_CUDA)
#include "../common/common.h"  // for AssertGPUSupport
#endif

namespace xgboost {
namespace cuda_impl {
void CopyGrad(Context const*, ArrayInterface<2, false> const&, ArrayInterface<2, false> const&,
              linalg::Matrix<GradientPair>*);

#if !defined(XGBOOST_USE_CUDA)
void CopyGrad(Context const*, ArrayInterface<2, false> const&, ArrayInterface<2, false> const&,
              linalg::Matrix<GradientPair>*) {
  common::AssertGPUSupport();
}
#endif
}  // namespace cuda_impl

namespace cpu_impl {
void CopyGrad(Context const* ctx, ArrayInterface<2, false> const& i_grad,
              ArrayInterface<2, false> const& i_hess, linalg::Matrix<GradientPair>* out_gpair) {
  out_gpair->Reshape(i_grad.Shape<0>(), i_grad.Shape<1>());
  auto h_gpair = out_gpair->HostView();
  DispatchDType(i_grad, DeviceOrd::CPU(), [&](auto&& t_grad) {
    DispatchDType(i_hess, DeviceOrd::CPU(), [&](auto&& t_hess) {
      common::ParallelFor(h_gpair.Size(), ctx->Threads(),
                          CustomGradHessOp{t_grad, t_hess, h_gpair});
    });
  });
}
}  // namespace cpu_impl

namespace {
void DispatchCopyGrad(Context const* ctx, ArrayInterface<2, false> const& i_grad,
                      ArrayInterface<2, false> const& i_hess,
                      linalg::Matrix<GradientPair>* out_gpair, data::ArrayCacheWriter* writer) {
  auto grad_is_cuda = ArrayInterfaceHandler::IsCudaPtr(i_grad.data);
  auto hess_is_cuda = ArrayInterfaceHandler::IsCudaPtr(i_hess.data);
  CHECK_EQ(grad_is_cuda, hess_is_cuda) << "gradient and hessian should be on the same device.";

  StringView msg{"Mismatched shape between the gradient and hessian."};
  CHECK_EQ(i_grad.Shape<0>(), i_hess.Shape<0>()) << msg;
  CHECK_EQ(i_grad.Shape<1>(), i_hess.Shape<1>()) << msg;

  auto gpair = std::make_shared<data::ArrayPage>();
  if (!grad_is_cuda) {
    cpu_impl::CopyGrad(ctx, i_grad, i_hess, &gpair->gpairs);
  } else {
    cuda_impl::CopyGrad(ctx, i_grad, i_hess, &gpair->gpairs);
  }

  // fixme
  linalg::Stack(out_gpair, gpair->gpairs);
  if (writer) {
    writer->Push(std::move(gpair));
  }
}
}  // namespace

GradientContainer::GradientContainer(Context const* ctx, common::Span<std::size_t const, 2> shape)
    : writer_{std::make_shared<data::ArrayCacheWriter>(ctx, shape)} {}

void GradientContainer::PushGrad(Context const* ctx, StringView grad, StringView hess) {
  ArrayInterface<2, false> i_grad{StringView{grad}};
  ArrayInterface<2, false> i_hess{StringView{hess}};
  DispatchCopyGrad(ctx, i_grad, i_hess, &gpair, this->writer_.get());
  if (writer_ && writer_->CanCommit()) {
    auto cache = this->writer_->Commit();
    this->writer_.reset();
    auto n_targets = cache;

    // fixme: cleanup
    std::map<std::string, std::shared_ptr<data::Cache>> cache_info;
    std::string cache_prefix = "grad";
    auto id = MakeCache(this, ".ap", true, cache_prefix, &cache_info);

    this->reader_ = std::make_shared<data::ArrayPageSource>(ctx, std::move(cache), cache_info.at(id));
  }
}

void GradientContainer::PushValueGrad(Context const* ctx, StringView grad, StringView hess) {
  ArrayInterface<2, false> i_grad{StringView{grad}};
  ArrayInterface<2, false> i_hess{StringView{hess}};
  // fixme: different writer
  DispatchCopyGrad(ctx, i_grad, i_hess, &value_gpair, nullptr);
}

BatchSet<data::ArrayPage> GradientContainer::GetGrad() {
  CHECK(this->reader_);
  return BatchSet{BatchIterator<data::ArrayPage>{this->reader_}};
}
}  // namespace xgboost
