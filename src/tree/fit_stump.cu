/**
 * Copyright 2022 by XGBoost Contributors
 *
 * \brief Utilities for estimating initial score.
 */
#if !defined(NOMINMAX) && defined(_WIN32)
#define NOMINMAX
#endif                                            // !defined(NOMINMAX)
#include <thrust/execution_policy.h>              // cuda::par
#include <thrust/iterator/counting_iterator.h>    // thrust::make_counting_iterator

#include <cstddef>                                // std::size_t

#include "../collective/device_communicator.cuh"  // DeviceCommunicator
#include "../common/device_helpers.cuh"           // dh::MakeTransformIterator
#include "fit_stump.h"
#include "xgboost/base.h"     // GradientPairPrecise, GradientPair, XGBOOST_DEVICE
#include "xgboost/context.h"  // Context
#include "xgboost/linalg.h"   // TensorView, Tensor, Constant
#include "xgboost/logging.h"  // CHECK_EQ
#include "xgboost/span.h"     // span

namespace xgboost {
namespace tree {
namespace cuda_impl {
void FitStump(Context const* ctx, linalg::TensorView<GradientPair const, 2> gpair,
              linalg::VectorView<float> out) {
  auto n_targets = out.Size();
  CHECK_EQ(n_targets, gpair.Shape(1));
  linalg::Vector<GradientPairPrecise> sum = linalg::Constant(ctx, GradientPairPrecise{}, n_targets);
  CHECK(out.Contiguous());

  // Reduce by column
  auto key_it = dh::MakeTransformIterator<bst_target_t>(
      thrust::make_counting_iterator(0ul),
      [=] XGBOOST_DEVICE(std::size_t i) -> bst_target_t { return i / gpair.Shape(0); });
  auto grad_it = dh::MakeTransformIterator<GradientPairPrecise>(
      thrust::make_counting_iterator(0ul),
      [=] XGBOOST_DEVICE(std::size_t i) -> GradientPairPrecise {
        auto target = i / gpair.Shape(0);
        auto sample = i % gpair.Shape(0);
        return GradientPairPrecise{gpair(sample, target)};
      });
  auto d_sum = sum.View(ctx->gpu_id);
  CHECK(d_sum.CContiguous());

  dh::XGBCachingDeviceAllocator<char> alloc;
  auto policy = thrust::cuda::par(alloc);
  thrust::reduce_by_key(policy, key_it, key_it + gpair.Size(), grad_it,
                        thrust::make_discard_iterator(), dh::tbegin(d_sum.Values()));

  collective::DeviceCommunicator* communicator = collective::Communicator::GetDevice(ctx->gpu_id);
  communicator->AllReduceSum(reinterpret_cast<double*>(d_sum.Values().data()), d_sum.Size() * 2);

  thrust::for_each_n(policy, thrust::make_counting_iterator(0ul), n_targets,
                     [=] XGBOOST_DEVICE(std::size_t i) mutable {
                       out(i) = static_cast<float>(
                           CalcUnregularizedWeight(d_sum(i).GetGrad(), d_sum(i).GetHess()));
                     });
}
}  // namespace cuda_impl
}  // namespace tree
}  // namespace xgboost
