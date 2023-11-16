/**
 * Copyright 2023, XGBoost Contributors
 */
#include <thrust/shuffle.h>  // for shuffle

#include <memory>  // for shared_ptr

#include "algorithm.cuh"     // for ArgSort
#include "cuda_context.cuh"  // for CUDAContext
#include "device_helpers.cuh"
#include "random.h"
#include "xgboost/base.h"                // for bst_feature_t
#include "xgboost/context.h"             // for Context
#include "xgboost/host_device_vector.h"  // for HostDeviceVector

namespace xgboost::common::cuda_impl {
// GPU implementation for sampling without replacement, see the CPU version for references.
void WeightedSamplingWithoutReplacement(Context const *ctx, common::Span<bst_feature_t const> array,
                                        common::Span<float const> weights,
                                        common::Span<bst_feature_t> results,
                                        HostDeviceVector<bst_feature_t> *sorted_idx,
                                        GlobalRandomEngine *grng) {
  CUDAContext const *cuctx = ctx->CUDACtx();
  CHECK_EQ(array.size(), weights.size());
  // Sampling keys
  dh::caching_device_vector<float> keys(weights.size());

  auto d_keys = dh::ToSpan(keys);

  auto seed = (*grng)();
  constexpr auto kEps = kRtEps;  // avoid CUDA compilation error
  thrust::for_each_n(cuctx->CTP(), thrust::make_counting_iterator(0ul), array.size(),
                     [=] XGBOOST_DEVICE(std::size_t i) {
                       thrust::default_random_engine rng;
                       rng.seed(seed);
                       rng.discard(i);
                       thrust::uniform_real_distribution<float> dist;

                       auto w = std::max(weights[i], kEps);
                       auto u = dist(rng);
                       auto k = std::log(u) / w;
                       d_keys[i] = k;
                     });
  // Allocate buffer for sorted index.
  auto d_idx = dh::LazyResize(ctx, sorted_idx, keys.size());

  ArgSort<false>(ctx, d_keys, d_idx);

  // Filter the result according to sorted index.
  auto it = thrust::make_permutation_iterator(dh::tbegin(array), dh::tbegin(d_idx));
  // |array| == |weights| == |keys| == |sorted_idx| >= |results|
  for (auto size : {array.size(), weights.size(), keys.size()}) {
    CHECK_EQ(size, d_idx.size());
  }
  CHECK_GE(array.size(), results.size());
  thrust::copy_n(cuctx->CTP(), it, results.size(), dh::tbegin(results));
}

void SampleFeature(Context const *ctx, bst_feature_t n_features,
                   std::shared_ptr<HostDeviceVector<bst_feature_t>> p_features,
                   std::shared_ptr<HostDeviceVector<bst_feature_t>> p_new_features,
                   HostDeviceVector<float> const &feature_weights,
                   HostDeviceVector<float> *weight_buffer,
                   HostDeviceVector<bst_feature_t> *idx_buffer, GlobalRandomEngine *grng) {
  CUDAContext const *cuctx = ctx->CUDACtx();
  auto &new_features = *p_new_features;
  new_features.SetDevice(ctx->Device());
  p_features->SetDevice(ctx->Device());
  CHECK_LE(n_features, p_features->Size());

  if (!feature_weights.Empty()) {
    CHECK_LE(p_features->Size(), feature_weights.Size());
    idx_buffer->SetDevice(ctx->Device());
    feature_weights.SetDevice(ctx->Device());

    auto d_old_features = p_features->DeviceSpan();
    auto d_weight_buffer = dh::LazyResize(ctx, weight_buffer, d_old_features.size());
    // Filter weights according to the existing feature index.
    auto d_feature_weight = feature_weights.ConstDeviceSpan();
    auto it = thrust::make_permutation_iterator(dh::tcbegin(d_feature_weight),
                                                dh::tcbegin(d_old_features));
    thrust::copy_n(cuctx->CTP(), it, d_old_features.size(), dh::tbegin(d_weight_buffer));
    new_features.Resize(n_features);
    WeightedSamplingWithoutReplacement(ctx, d_old_features, d_weight_buffer,
                                       new_features.DeviceSpan(), idx_buffer, grng);
  } else {
    new_features.Resize(p_features->Size());
    new_features.Copy(*p_features);
    auto d_feat = new_features.DeviceSpan();
    thrust::default_random_engine rng;
    rng.seed((*grng)());
    thrust::shuffle(cuctx->CTP(), dh::tbegin(d_feat), dh::tend(d_feat), rng);
    new_features.Resize(n_features);
  }

  auto d_new_features = new_features.DeviceSpan();
  thrust::sort(cuctx->CTP(), dh::tbegin(d_new_features), dh::tend(d_new_features));
}

void InitFeatureSet(Context const *ctx,
                    std::shared_ptr<HostDeviceVector<bst_feature_t>> p_features) {
  CUDAContext const *cuctx = ctx->CUDACtx();
  auto d_features = p_features->DeviceSpan();
  thrust::sequence(cuctx->CTP(), dh::tbegin(d_features), dh::tend(d_features), 0);
}
}  // namespace xgboost::common::cuda_impl
