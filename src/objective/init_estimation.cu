#include <thrust/iterator/counting_iterator.h>  // thrust::make_counting_iterator

#include "../common/device_helpers.cuh"  // dh::MakeTransformIterator
#include "../common/numeric.cuh"         // Reduce
#include "init_estimation.h"

namespace xgboost {
namespace obj {
namespace cuda_impl {
double WeightedMean(Context const* ctx, MetaInfo const& info) {
  std::uint64_t n_samples = info.num_row_;
  auto y = info.labels.View(ctx->gpu_id);
  auto w = common::OptionalWeights{info.weights_.ConstHostSpan()};
  auto it = dh::MakeTransformIterator<double>(
      thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(size_t i) -> double {
        size_t r, c;
        std::tie(r, c) = linalg::UnravelIndex(i, y.Shape());
        return y(r, c) * w[r] / static_cast<double>(n_samples);
      });
  return common::cuda_impl::Reduce(ctx, it, it + y.Size(), 0.0);
}
}  // namespace cuda_impl
}  // namespace obj
}  // namespace xgboost
