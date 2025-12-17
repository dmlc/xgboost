#include "../common/cuda_context.cuh"
#include "../common/cuda_rt_utils.h"
#include "xgboost/context.h"
#include "xgboost/span.h"

namespace xgboost::data {
namespace cuda_impl {
void ReadArrayPage(Context const* ctx, common::Span<GradientPair> d_dst,
                   common::Span<GradientPair const> h_src) {
  curt::MemcpyAsync(d_dst.data(), h_src.data(), d_dst.size_bytes(), ctx->CUDACtx()->Stream());
  ctx->CUDACtx()->Stream().Sync();
}
}  // namespace cuda_impl
}  // namespace xgboost::data
