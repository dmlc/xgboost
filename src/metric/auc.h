/*!
 * Copyright 2021 by XGBoost Contributors
 */
#ifndef XGBOOST_METRIC_AUC_H_
#define XGBOOST_METRIC_AUC_H_
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <tuple>
#include <utility>

#include "../collective/communicator-inl.h"
#include "../common/common.h"
#include "../common/threading_utils.h"
#include "xgboost/base.h"
#include "xgboost/data.h"
#include "xgboost/metric.h"
#include "xgboost/span.h"

namespace xgboost {
namespace metric {
/***********
 * ROC AUC *
 ***********/
XGBOOST_DEVICE inline double TrapezoidArea(double x0, double x1, double y0, double y1) {
  return std::abs(x0 - x1) * (y0 + y1) * 0.5f;
}

struct DeviceAUCCache;

std::tuple<double, double, double> GPUBinaryROCAUC(common::Span<float const> predts,
                                                   MetaInfo const &info, std::int32_t device,
                                                   std::shared_ptr<DeviceAUCCache> *p_cache);

double GPUMultiClassROCAUC(Context const *ctx, common::Span<float const> predts,
                           MetaInfo const &info, std::shared_ptr<DeviceAUCCache> *p_cache,
                           std::size_t n_classes);

std::pair<double, std::uint32_t> GPURankingAUC(Context const *ctx, common::Span<float const> predts,
                                               MetaInfo const &info,
                                               std::shared_ptr<DeviceAUCCache> *cache);

/**********
 * PR AUC *
 **********/
std::tuple<double, double, double> GPUBinaryPRAUC(common::Span<float const> predts,
                                                  MetaInfo const &info, std::int32_t device,
                                                  std::shared_ptr<DeviceAUCCache> *p_cache);

double GPUMultiClassPRAUC(Context const *ctx, common::Span<float const> predts,
                          MetaInfo const &info, std::shared_ptr<DeviceAUCCache> *p_cache,
                          std::size_t n_classes);

std::pair<double, std::uint32_t> GPURankingPRAUC(Context const *ctx,
                                                 common::Span<float const> predts,
                                                 MetaInfo const &info,
                                                 std::shared_ptr<DeviceAUCCache> *cache);

namespace detail {
XGBOOST_DEVICE inline double CalcH(double fp_a, double fp_b, double tp_a,
                                   double tp_b) {
  return (fp_b - fp_a) / (tp_b - tp_a);
}

XGBOOST_DEVICE inline double CalcB(double fp_a, double h, double tp_a, double total_pos) {
  return (fp_a - h * tp_a) / total_pos;
}

XGBOOST_DEVICE inline double CalcA(double h) { return h + 1; }

XGBOOST_DEVICE inline double CalcDeltaPRAUC(double fp_prev, double fp,
                                            double tp_prev, double tp,
                                            double total_pos) {
  double pr_prev = tp_prev / total_pos;
  double pr = tp / total_pos;

  double h{0}, a{0}, b{0};

  if (tp == tp_prev) {
    a = 1.0;
    b = 0.0;
  } else {
    h = detail::CalcH(fp_prev, fp, tp_prev, tp);
    a = detail::CalcA(h);
    b = detail::CalcB(fp_prev, h, tp_prev, total_pos);
  }

  double area = 0;
  if (b != 0.0) {
    area = (pr - pr_prev -
            b / a * (std::log(a * pr + b) - std::log(a * pr_prev + b))) /
           a;
  } else {
    area = (pr - pr_prev) / a;
  }
  return area;
}
}  // namespace detail

inline void InvalidGroupAUC() {
  LOG(INFO) << "Invalid group with less than 3 samples is found on worker "
            << collective::GetRank() << ".  Calculating AUC value requires at "
            << "least 2 pairs of samples.";
}

struct PRAUCLabelInvalid {
  XGBOOST_DEVICE bool operator()(float y) { return y < 0.0f || y > 1.0f; }
};

inline void InvalidLabels() {
  LOG(FATAL) << "PR-AUC supports only binary relevance for learning to rank.";
}
}      // namespace metric
}      // namespace xgboost
#endif  // XGBOOST_METRIC_AUC_H_
