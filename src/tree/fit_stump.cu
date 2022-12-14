/**
 * Copyright 2022 by XGBoost Contributors
 *
 * \brief Utilities for estimating initial score.
 */
#if !defined(NOMINMAX) && defined(_WIN32)
#define NOMINMAX
#endif                                            // !defined(NOMINMAX)
#include <thrust/execution_policy.h>              // cuda::par
#include <thrust/functional.h>                    // thrust::equal_to
#include <thrust/iterator/counting_iterator.h>    // thrust::make_counting_iterator
#include <thrust/iterator/zip_iterator.h>         // thrust::make_zip_iterator

#include <algorithm>                              // std::max
#include <cstddef>                                // std::size_t

#include "../collective/device_communicator.cuh"  // DeviceCommunicator
#include "../common/device_helpers.cuh"           // dh::MakeTransformIterator::Reduce,TypedDiscard
#include "fit_stump.h"
#include "xgboost/base.h"     // GradientPairPrecise, GradientPair, XGBOOST_DEVICE
#include "xgboost/context.h"  // Context
#include "xgboost/span.h"     // span

namespace xgboost {
namespace obj {
namespace cuda_impl {
void FitStump(Context const* ctx, linalg::TensorView<GradientPair const, 2> gpair,
              linalg::VectorView<float> out) {
  // 2 rows, first one is gradient, sencond one is hessian. Number of columns equal to
  // number of targets.
  auto n_targets = out.Size();
  CHECK_EQ(n_targets, gpair.Shape(1));
  linalg::Tensor<double, 2> sum = linalg::Zeros<double>(ctx, 2, n_targets);
  CHECK(out.Contiguous());
  auto sum_grad = sum.View(ctx->gpu_id).Slice(0, linalg::All());
  auto sum_hess = sum.View(ctx->gpu_id).Slice(1, linalg::All());

  // Reduce by column
  auto key_it = dh::MakeTransformIterator<bst_target_t>(
      thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(std::size_t i) -> bst_target_t {
        return i / gpair.Shape(0);
        return std::get<1>(linalg::UnravelIndex(i, gpair.Shape()));
      });
  auto grad_it = dh::MakeTransformIterator<double>(thrust::make_counting_iterator(0ul),
                                                   [=] XGBOOST_DEVICE(std::size_t i) -> double {
                                                     auto target = i / gpair.Shape(0);
                                                     auto sample = i % gpair.Shape(0);
                                                     return gpair(sample, target).GetGrad();
                                                   });
  auto hess_it = dh::MakeTransformIterator<double>(thrust::make_counting_iterator(0ul),
                                                   [=] XGBOOST_DEVICE(std::size_t i) -> double {
                                                     auto target = i / gpair.Shape(0);
                                                     auto sample = i % gpair.Shape(0);
                                                     return gpair(sample, target).GetHess();
                                                   });
  auto val_it = thrust::make_zip_iterator(grad_it, hess_it);
  CHECK(sum_grad.CContiguous());
  CHECK(sum_hess.CContiguous());
  auto out_it =
      thrust::make_zip_iterator(dh::tbegin(sum_grad.Values()), dh::tbegin(sum_hess.Values()));

  dh::XGBCachingDeviceAllocator<char> alloc;
  auto policy = thrust::cuda::par(alloc);
  thrust::reduce_by_key(policy, key_it, key_it + gpair.Size(), val_it,
                        dh::TypedDiscard<bst_target_t>{}, out_it, thrust::equal_to<bst_target_t>{},
                        [=] __device__(auto lhs, auto rhs) {
                          return thrust::make_tuple(thrust::get<0>(lhs) + thrust::get<0>(rhs),
                                                    thrust::get<1>(lhs) + thrust::get<1>(rhs));
                        });

  collective::DeviceCommunicator* communicator = collective::Communicator::GetDevice(ctx->gpu_id);
  communicator->AllReduceSum(sum_grad.Values().data(), sum_grad.Size());
  communicator->AllReduceSum(sum_hess.Values().data(), sum_hess.Size());

  thrust::for_each_n(policy, thrust::make_counting_iterator(0ul), n_targets,
                     [=] XGBOOST_DEVICE(std::size_t i) mutable {
                       out(i) = static_cast<float>(CalcUnregulatedWeight(sum_grad(i), sum_hess(i)));
                     });
}
}  // namespace cuda_impl
}  // namespace obj
}  // namespace xgboost
