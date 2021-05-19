/*!
 * Copyright 2021 by XGBoost Contributors
 */
#ifndef XGBOOST_METRIC_AUC_H_
#define XGBOOST_METRIC_AUC_H_
#include <cmath>
#include <memory>
#include <tuple>
#include <utility>

#include "rabit/rabit.h"
#include "xgboost/base.h"
#include "xgboost/span.h"
#include "xgboost/data.h"

namespace xgboost {
namespace metric {
XGBOOST_DEVICE inline float TrapesoidArea(float x0, float x1, float y0, float y1) {
  return std::abs(x0 - x1) * (y0 + y1) * 0.5f;
}

struct DeviceAUCCache;

std::tuple<float, float, float>
GPUBinaryAUC(common::Span<float const> predts, MetaInfo const &info,
             int32_t device, std::shared_ptr<DeviceAUCCache> *p_cache);

float GPUMultiClassAUCOVR(common::Span<float const> predts, MetaInfo const &info,
                          int32_t device, std::shared_ptr<DeviceAUCCache>* cache,
                          size_t n_classes);

std::pair<float, uint32_t>
GPURankingAUC(common::Span<float const> predts, MetaInfo const &info,
              int32_t device, std::shared_ptr<DeviceAUCCache> *cache);

inline void InvalidGroupAUC() {
  LOG(INFO) << "Invalid group with less than 3 samples is found on worker "
            << rabit::GetRank() << ".  Calculating AUC value requires at "
            << "least 2 pairs of samples.";
}
}      // namespace metric
}      // namespace xgboost
#endif  // XGBOOST_METRIC_AUC_H_
