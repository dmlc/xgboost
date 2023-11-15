/**
 * Copyright 2023, XGBoost Contributors
 */
#include <thrust/shuffle.h>  // for shuffle

#include <memory>  // for shared_ptr

#include "cuda_context.cuh"  // for CUDAContext
#include "device_helpers.cuh"
#include "random.h"
#include "xgboost/base.h"                // for bst_feature_t
#include "xgboost/context.h"             // for Context
#include "xgboost/host_device_vector.h"  // for HostDeviceVector

namespace xgboost::common::cuda_impl {
void WeightedSamplingWithoutReplacement(Context const *ctx, common::Span<bst_feature_t const> array,
                                        common::Span<float const> weights,
                                        common::Span<bst_feature_t> results, std::size_t n,
                                        HostDeviceVector<bst_feature_t> *idx,
                                        GlobalRandomEngine &grng) {
  CUDAContext const *cuctx = ctx->CUDACtx();
  CHECK_EQ(array.size(), weights.size());
  dh::caching_device_vector<float> keys(weights.size());

  auto d_keys = dh::ToSpan(keys);

  auto seed = grng();
  constexpr auto kEps = kRtEps;
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
  idx->SetDevice(ctx->Device());
  idx->Resize(keys.size());

  dh::ArgSort<false>(d_keys, idx->DeviceSpan());
  auto it = thrust::make_permutation_iterator(dh::tbegin(array), dh::tbegin(idx->DeviceSpan()));
  CHECK_GE(results.size(), n);
  thrust::copy_n(it, n, dh::tbegin(results));
}

void SampleFeature(Context const *ctx, bst_feature_t n_features,
                   std::shared_ptr<HostDeviceVector<bst_feature_t>> p_features,
                   std::shared_ptr<HostDeviceVector<bst_feature_t>> p_new_features,
                   HostDeviceVector<float> const &feature_weights,
                   HostDeviceVector<float> *weight_buffer,
                   HostDeviceVector<bst_feature_t> *idx_buffer, GlobalRandomEngine &grng) {
  CUDAContext const *cuctx = ctx->CUDACtx();
  auto &new_features = *p_new_features;
  new_features.SetDevice(ctx->Device());
  p_features->SetDevice(ctx->Device());

  if (!feature_weights.Empty()) {
    weight_buffer->SetDevice(ctx->Device());
    idx_buffer->SetDevice(ctx->Device());
    feature_weights.SetDevice(ctx->Device());

    auto d_features = p_features->DeviceSpan();
    if (weight_buffer->Size() < feature_weights.Size()) {
      weight_buffer->Resize(feature_weights.Size());
    }
    auto d_weight = weight_buffer->DeviceSpan().subspan(0, d_features.size());
    auto d_feature_weight = feature_weights.ConstDeviceSpan();
    auto it =
        thrust::make_permutation_iterator(dh::tcbegin(d_feature_weight), dh::tcbegin(d_features));
    thrust::copy_n(cuctx->CTP(), it, d_features.size(), dh::tbegin(d_weight));
    new_features.Resize(n_features);
    WeightedSamplingWithoutReplacement(ctx, d_features, d_weight, new_features.DeviceSpan(),
                                       n_features, idx_buffer, grng);
  } else {
    new_features.Resize(p_features->Size());
    new_features.Copy(*p_features);
    auto d_feat = new_features.DeviceSpan();
    thrust::default_random_engine rng;
    rng.seed(grng());
    thrust::shuffle(cuctx->CTP(), dh::tbegin(d_feat), dh::tend(d_feat), rng);
    new_features.Resize(n_features);
  }

  auto d_new_features = new_features.DeviceSpan();
  thrust::sort(dh::tbegin(d_new_features), dh::tend(d_new_features));
}
}  // namespace xgboost::common::cuda_impl
