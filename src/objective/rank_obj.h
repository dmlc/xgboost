/*!
 * Copyright 2021 XGBoost contributors
 */
#ifndef XGBOOST_OBJECTIVE_RANK_OBJ_H_
#define XGBOOST_OBJECTIVE_RANK_OBJ_H_
#include <algorithm>
#include <memory>
#include <utility>

#include "../common/math.h"
#include "../common/ranking_utils.h"
#include "xgboost/base.h"
#include "xgboost/data.h"
#include "xgboost/parameter.h"
#include "xgboost/span.h"

#if defined(__CUDACC__)
#include "../common/device_helpers.cuh"
#endif  // defined(__CUDACC__)

namespace xgboost {
namespace obj {
XGBOOST_DEVICE inline float DeltaNDCG(float s_high, float s_low, uint32_t y_high, uint32_t y_low,
                                      uint32_t r_high, uint32_t r_low, float inv_IDCG) {
  float gain_high = ::xgboost::CalcNDCGGain(y_high);
  float discount_high = CalcNDCGDiscount(r_high);

  float gain_low = CalcNDCGGain(y_low);
  float discount_low = CalcNDCGDiscount(r_low);
  bst_float original = gain_high * discount_high + gain_low * discount_low;
  float changed = gain_low * discount_high + gain_high * discount_low;

  bst_float delta_NDCG = (original - changed) * inv_IDCG;
  return delta_NDCG;
}

XGBOOST_DEVICE inline void LambdaNDCG(common::Span<float const> labels,
                                      common::Span<float const> predts,
                                      common::Span<size_t const> sorted_idx, size_t i, size_t j,
                                      float inv_IDCG, common::Span<GradientPair> gpairs) {
  if (labels[sorted_idx[i]] == labels[sorted_idx[j]]) {
    return;
  }
  size_t rank_high = i, rank_low = j;
  if (labels[sorted_idx[i]] <= labels[sorted_idx[j]]) {
#if defined(__CUDACC__)
    thrust::swap(rank_high, rank_low);
#else
    std::swap(rank_high, rank_low);
#endif  // defined(__CUDACC__)
  }
  size_t idx_high = sorted_idx[rank_high];
  size_t idx_low = sorted_idx[rank_low];

  uint32_t y_high = static_cast<uint32_t>(labels[idx_high]);
  float s_high = predts[idx_high];
  uint32_t y_low = static_cast<uint32_t>(labels[idx_low]);
  float s_low = predts[idx_low];

  float sigmoid = common::Sigmoid(s_high - s_low);
  float delta_NDCG = fabs(DeltaNDCG(s_high, s_low, y_high, y_low, rank_high, rank_low, inv_IDCG));
  float lambda_ij = (sigmoid - 1.0f) * delta_NDCG;
  constexpr float kEps = 1e-16f;
  float hessian_ij = (std::max(sigmoid * (1.0f - sigmoid), kEps)) * delta_NDCG;

#if defined(__CUDA_ARCH__)
  dh::AtomicAddGpair(gpairs.data() + idx_high, GradientPair{lambda_ij, hessian_ij});
  dh::AtomicAddGpair(gpairs.data() + idx_low, GradientPair{-lambda_ij, hessian_ij});
#else
  gpairs[idx_high] += GradientPair{lambda_ij, hessian_ij};
  gpairs[idx_low] += GradientPair{-lambda_ij, hessian_ij};
#endif
}

struct LambdaMARTParam : public XGBoostParameter<LambdaMARTParam> {
  size_t lambdamart_truncation;

  DMLC_DECLARE_PARAMETER(LambdaMARTParam) {
    DMLC_DECLARE_FIELD(lambdamart_truncation).set_default(1).describe("");
  }
};

void CheckNDCGLabelsCPUKernel(LambdaMARTParam const& p, common::Span<float const> labels);
void CheckNDCGLabelsGPUKernel(LambdaMARTParam const& p, common::Span<float const> labels);

struct DeviceNDCGCache;

void LambdaMARTGetGradientNDCGGPUKernel(const HostDeviceVector<bst_float>& preds,
                                        const MetaInfo& info, size_t ndcg_truncation,
                                        std::shared_ptr<DeviceNDCGCache>* cache, int32_t device_id,
                                        HostDeviceVector<GradientPair>* out_gpair);
}  // namespace obj
}  // namespace xgboost
#endif  // XGBOOST_OBJECTIVE_RANK_OBJ_H_
