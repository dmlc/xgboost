/*!
 * Copyright 2021 XGBoost contributors
 */
#ifndef XGBOOST_COMMON_RANKING_UTILS_H_
#define XGBOOST_COMMON_RANKING_UTILS_H_

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <vector>

#include "xgboost/base.h"
#include "xgboost/logging.h"

namespace xgboost {
XGBOOST_DEVICE inline float CalcNDCGGain(uint32_t label) {
  label = std::min(31u, label);
  return (1u << label) - 1;
}

XGBOOST_DEVICE inline float CalcNDCGDiscount(size_t idx) {
  return 1.0f / std::log2(static_cast<float>(idx) + 2.0f);
}

template <typename Container>
inline float CalcDCGAtK(Container const& scores, size_t k) {
  double sumdcg = 0;
  for (size_t i = 0; i < k; ++i) {
    float gain = CalcNDCGGain(scores[i]);
    float discount = CalcNDCGDiscount(i);
    sumdcg += gain * discount;
  }
  return sumdcg;
}

template <typename Container>
float CalcInvIDCG(Container const& sorted_labels, size_t p) {
  double sumdcg = 0;
  for (size_t i = 0; i < p && i < sorted_labels.size(); ++i) {
    float gain = CalcNDCGGain(sorted_labels[i]);
    float discount = CalcNDCGDiscount(i);
    sumdcg += gain * discount;
  }
  sumdcg = sumdcg == 0.0f ? 0.0f : 1.0 / sumdcg;
  return sumdcg;
}

template <typename Container>
inline float CalcNDCGAtK(Container const& scores, Container const& labels, size_t k) {
  float dcg = CalcDCGAtK(scores, k);
  float idcg = CalcDCGAtK(labels, k);
  return idcg == 0 ? 0 : dcg / idcg;
}
}  // namespace xgboost
#endif  // XGBOOST_COMMON_RANKING_UTILS_H_
