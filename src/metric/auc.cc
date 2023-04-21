/**
 * Copyright 2021-2023 by XGBoost Contributors
 */
#include "auc.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include "../common/algorithm.h"        // ArgSort
#include "../common/math.h"
#include "../common/optional_weight.h"  // OptionalWeights
#include "metric_common.h"              // MetricNoCache
#include "xgboost/context.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/linalg.h"
#include "xgboost/metric.h"

namespace xgboost {
namespace metric {
// tag the this file, used by force static link later.
DMLC_REGISTRY_FILE_TAG(auc);
/**
 * Calculate AUC for binary classification problem.  This function does not normalize the
 * AUC by 1 / (num_positive * num_negative), instead it returns a tuple for caller to
 * handle the normalization.
 */
template <typename Fn>
std::tuple<double, double, double>
BinaryAUC(common::Span<float const> predts, linalg::VectorView<float const> labels,
          common::OptionalWeights weights,
          std::vector<size_t> const &sorted_idx, Fn &&area_fn) {
  CHECK_NE(labels.Size(), 0);
  CHECK_EQ(labels.Size(), predts.size());
  auto p_predts = predts.data();

  double auc{0};

  float label = labels(sorted_idx.front());
  float w = weights[sorted_idx[0]];
  double fp = (1.0 - label) * w, tp = label * w;
  double tp_prev = 0, fp_prev = 0;
  // TODO(jiaming): We can parallize this if we have a parallel scan for CPU.
  for (size_t i = 1; i < sorted_idx.size(); ++i) {
    if (p_predts[sorted_idx[i]] != p_predts[sorted_idx[i - 1]]) {
      auc += area_fn(fp_prev, fp, tp_prev, tp);
      tp_prev = tp;
      fp_prev = fp;
    }
    label = labels(sorted_idx[i]);
    float w = weights[sorted_idx[i]];
    fp += (1.0f - label) * w;
    tp += label * w;
  }

  auc += area_fn(fp_prev, fp, tp_prev, tp);
  if (fp <= 0.0f || tp <= 0.0f) {
    auc = 0;
    fp = 0;
    tp = 0;
  }

  return std::make_tuple(fp, tp, auc);
}

/**
 * Calculate AUC for multi-class classification problem using 1-vs-rest approach.
 *
 * TODO(jiaming): Use better algorithms like:
 *
 * - Kleiman, Ross and Page, David. $AUC_{\mu}$: A Performance Metric for Multi-Class
 *   Machine Learning Models
 */
template <typename BinaryAUC>
double MultiClassOVR(Context const *ctx, common::Span<float const> predts, MetaInfo const &info,
                     size_t n_classes, int32_t n_threads, BinaryAUC &&binary_auc) {
  CHECK_NE(n_classes, 0);
  auto const labels = info.labels.View(Context::kCpuId);
  if (labels.Shape(0) != 0) {
    CHECK_EQ(labels.Shape(1), 1) << "AUC doesn't support multi-target model.";
  }

  std::vector<double> results_storage(n_classes * 3, 0);
  linalg::TensorView<double, 2> results(results_storage, {n_classes, static_cast<size_t>(3)},
                                        Context::kCpuId);
  auto local_area = results.Slice(linalg::All(), 0);
  auto tp = results.Slice(linalg::All(), 1);
  auto auc = results.Slice(linalg::All(), 2);

  auto weights = common::OptionalWeights{info.weights_.ConstHostSpan()};
  auto predts_t = linalg::TensorView<float const, 2>(
      predts, {static_cast<size_t>(info.num_row_), n_classes},
      Context::kCpuId);

  if (info.labels.Size() != 0) {
    common::ParallelFor(n_classes, n_threads, [&](auto c) {
      std::vector<float> proba(info.labels.Size());
      std::vector<float> response(info.labels.Size());
      for (size_t i = 0; i < proba.size(); ++i) {
        proba[i] = predts_t(i, c);
        response[i] = labels(i) == c ? 1.0f : 0.0;
      }
      double fp;
      std::tie(fp, tp(c), auc(c)) =
          binary_auc(ctx, proba, linalg::MakeVec(response.data(), response.size(), -1), weights);
      local_area(c) = fp * tp(c);
    });
  }

  // we have 2 averages going in here, first is among workers, second is among
  // classes. allreduce sums up fp/tp auc for each class.
  collective::GlobalSum(info, &results.Values());
  double auc_sum{0};
  double tp_sum{0};
  for (size_t c = 0; c < n_classes; ++c) {
    if (local_area(c) != 0) {
      // normalize and weight it by prevalence.  After allreduce, `local_area`
      // means the total covered area (not area under curve, rather it's the
      // accessible area for each worker) for each class.
      auc_sum += auc(c) / local_area(c) * tp(c);
      tp_sum += tp(c);
    } else {
      auc_sum = std::numeric_limits<double>::quiet_NaN();
      break;
    }
  }
  if (tp_sum == 0 || std::isnan(auc_sum)) {
    auc_sum = std::numeric_limits<double>::quiet_NaN();
  } else {
    auc_sum /= tp_sum;
  }
  return auc_sum;
}

std::tuple<double, double, double> BinaryROCAUC(Context const *ctx,
                                                common::Span<float const> predts,
                                                linalg::VectorView<float const> labels,
                                                common::OptionalWeights weights) {
  auto const sorted_idx =
      common::ArgSort<size_t>(ctx, predts.data(), predts.data() + predts.size(), std::greater<>{});
  return BinaryAUC(predts, labels, weights, sorted_idx, TrapezoidArea);
}

/**
 * Calculate AUC for 1 ranking group;
 */
double GroupRankingROC(Context const* ctx, common::Span<float const> predts,
                       linalg::VectorView<float const> labels, float w) {
  // on ranking, we just count all pairs.
  double auc{0};
  // argsort doesn't support tensor input yet.
  auto raw_labels = labels.Values().subspan(0, labels.Size());
  auto const sorted_idx = common::ArgSort<size_t>(
      ctx, raw_labels.data(), raw_labels.data() + raw_labels.size(), std::greater<>{});
  w = common::Sqr(w);

  double sum_w = 0.0f;
  for (size_t i = 0; i < labels.Size(); ++i) {
    for (size_t j = i + 1; j < labels.Size(); ++j) {
      auto predt = predts[sorted_idx[i]] - predts[sorted_idx[j]];
      if (predt > 0) {
        predt = 1.0;
      } else if (predt == 0) {
        predt = 0.5;
      } else {
        predt = 0;
      }
      auc += predt * w;
      sum_w += w;
    }
  }
  if (sum_w != 0) {
    auc /= sum_w;
  }
  CHECK_LE(auc, 1.0f);
  return auc;
}

/**
 * \brief PR-AUC for binary classification.
 *
 *   https://doi.org/10.1371/journal.pone.0092209
 */
std::tuple<double, double, double> BinaryPRAUC(Context const *ctx, common::Span<float const> predts,
                                               linalg::VectorView<float const> labels,
                                               common::OptionalWeights weights) {
  auto const sorted_idx =
      common::ArgSort<size_t>(ctx, predts.data(), predts.data() + predts.size(), std::greater<>{});
  double total_pos{0}, total_neg{0};
  for (size_t i = 0; i < labels.Size(); ++i) {
    auto w = weights[i];
    total_pos += w * labels(i);
    total_neg += w * (1.0f - labels(i));
  }
  if (total_pos <= 0 || total_neg <= 0) {
    return {1.0f, 1.0f, std::numeric_limits<float>::quiet_NaN()};
  }
  auto fn = [total_pos](double fp_prev, double fp, double tp_prev, double tp) {
    return detail::CalcDeltaPRAUC(fp_prev, fp, tp_prev, tp, total_pos);
  };

  double tp{0}, fp{0}, auc{0};
  std::tie(fp, tp, auc) = BinaryAUC(predts, labels, weights, sorted_idx, fn);
  return std::make_tuple(1.0, 1.0, auc);
}

/**
 * Cast LTR problem to binary classification problem by comparing pairs.
 */
template <bool is_roc>
std::pair<double, uint32_t> RankingAUC(Context const *ctx, std::vector<float> const &predts,
                                       MetaInfo const &info, int32_t n_threads) {
  CHECK_GE(info.group_ptr_.size(), 2);
  uint32_t n_groups = info.group_ptr_.size() - 1;
  auto s_predts = common::Span<float const>{predts};
  auto labels = info.labels.View(Context::kCpuId);
  auto s_weights = info.weights_.ConstHostSpan();

  std::atomic<uint32_t> invalid_groups{0};

  std::vector<double> auc_tloc(n_threads, 0);
  common::ParallelFor(n_groups, n_threads, [&](size_t g) {
    g += 1;  // indexing needs to start from 1
    size_t cnt = info.group_ptr_[g] - info.group_ptr_[g - 1];
    float w = s_weights.empty() ? 1.0f : s_weights[g - 1];
    auto g_predts = s_predts.subspan(info.group_ptr_[g - 1], cnt);
    auto g_labels = labels.Slice(linalg::Range(info.group_ptr_[g - 1], info.group_ptr_[g]));
    double auc;
    if (is_roc && g_labels.Size() < 3) {
      // With 2 documents, there's only 1 comparison can be made.  So either
      // TP or FP will be zero.
      invalid_groups++;
      auc = 0;
    } else {
      if (is_roc) {
        auc = GroupRankingROC(ctx, g_predts, g_labels, w);
      } else {
        auc = std::get<2>(BinaryPRAUC(ctx, g_predts, g_labels, common::OptionalWeights{w}));
      }
      if (std::isnan(auc)) {
        invalid_groups++;
        auc = 0;
      }
    }
    auc_tloc[omp_get_thread_num()] += auc;
  });
  double sum_auc = std::accumulate(auc_tloc.cbegin(), auc_tloc.cend(), 0.0);

  return std::make_pair(sum_auc, n_groups - invalid_groups);
}

template <typename Curve>
class EvalAUC : public MetricNoCache {
  double Eval(const HostDeviceVector<bst_float> &preds, const MetaInfo &info) override {
    double auc {0};
    if (ctx_->gpu_id != Context::kCpuId) {
      preds.SetDevice(ctx_->gpu_id);
      info.labels.SetDevice(ctx_->gpu_id);
      info.weights_.SetDevice(ctx_->gpu_id);
    }
    //  We use the global size to handle empty dataset.
    std::array<size_t, 2> meta{info.labels.Size(), preds.Size()};
    if (!info.IsVerticalFederated()) {
      collective::Allreduce<collective::Operation::kMax>(meta.data(), meta.size());
    }
    if (meta[0] == 0) {
      // Empty across all workers, which is not supported.
      auc = std::numeric_limits<double>::quiet_NaN();
    } else if (!info.group_ptr_.empty()) {
      /**
       * learning to rank
       */
      if (!info.weights_.Empty()) {
        CHECK_EQ(info.weights_.Size(), info.group_ptr_.size() - 1);
      }
      uint32_t valid_groups = 0;
      if (info.labels.Size() != 0) {
        CHECK_EQ(info.group_ptr_.back(), info.labels.Size());
        std::tie(auc, valid_groups) =
            static_cast<Curve *>(this)->EvalRanking(preds, info);
      }
      if (valid_groups != info.group_ptr_.size() - 1) {
        InvalidGroupAUC();
      }

      auc = collective::GlobalRatio(info, auc, static_cast<double>(valid_groups));
      if (!std::isnan(auc)) {
        CHECK_LE(auc, 1) << "Total AUC across groups: " << auc * valid_groups
                         << ", valid groups: " << valid_groups;
      }
    } else if (meta[0] != meta[1] && meta[1] % meta[0] == 0) {
      /**
       * multi class
       */
      size_t n_classes = meta[1] / meta[0];
      CHECK_NE(n_classes, 0);
      auc = static_cast<Curve *>(this)->EvalMultiClass(preds, info, n_classes);
    } else {
      /**
       * binary classification
       */
      double fp{0}, tp{0};
      if (!(preds.Empty() || info.labels.Size() == 0)) {
        std::tie(fp, tp, auc) =
            static_cast<Curve *>(this)->EvalBinary(preds, info);
      }
      auc = collective::GlobalRatio(info, auc, fp * tp);
      if (!std::isnan(auc)) {
        CHECK_LE(auc, 1.0);
      }
    }
    if (std::isnan(auc)) {
      LOG(WARNING) << "Dataset is empty, or contains only positive or negative samples.";
    }
    return auc;
  }
};

class EvalROCAUC : public EvalAUC<EvalROCAUC> {
  std::shared_ptr<DeviceAUCCache> d_cache_;

 public:
  std::pair<double, uint32_t> EvalRanking(HostDeviceVector<float> const &predts,
                                          MetaInfo const &info) {
    double auc{0};
    uint32_t valid_groups = 0;
    auto n_threads = ctx_->Threads();
    if (ctx_->gpu_id == Context::kCpuId) {
      std::tie(auc, valid_groups) =
          RankingAUC<true>(ctx_, predts.ConstHostVector(), info, n_threads);
    } else {
      std::tie(auc, valid_groups) =
          GPURankingAUC(ctx_, predts.ConstDeviceSpan(), info, &this->d_cache_);
    }
    return std::make_pair(auc, valid_groups);
  }

  double EvalMultiClass(HostDeviceVector<float> const &predts,
                        MetaInfo const &info, size_t n_classes) {
    double auc{0};
    auto n_threads = ctx_->Threads();
    CHECK_NE(n_classes, 0);
    if (ctx_->gpu_id == Context::kCpuId) {
      auc = MultiClassOVR(ctx_, predts.ConstHostVector(), info, n_classes, n_threads, BinaryROCAUC);
    } else {
      auc = GPUMultiClassROCAUC(ctx_, predts.ConstDeviceSpan(), info, &this->d_cache_, n_classes);
    }
    return auc;
  }

  std::tuple<double, double, double>
  EvalBinary(HostDeviceVector<float> const &predts, MetaInfo const &info) {
    double fp, tp, auc;
    if (ctx_->gpu_id == Context::kCpuId) {
      std::tie(fp, tp, auc) = BinaryROCAUC(ctx_, predts.ConstHostVector(),
                                           info.labels.HostView().Slice(linalg::All(), 0),
                                           common::OptionalWeights{info.weights_.ConstHostSpan()});
    } else {
      std::tie(fp, tp, auc) = GPUBinaryROCAUC(predts.ConstDeviceSpan(), info,
                                              ctx_->gpu_id, &this->d_cache_);
    }
    return std::make_tuple(fp, tp, auc);
  }

 public:
  char const* Name() const override {
    return "auc";
  }
};

XGBOOST_REGISTER_METRIC(EvalAUC, "auc")
.describe("Receiver Operating Characteristic Area Under the Curve.")
.set_body([](const char*) { return new EvalROCAUC(); });

#if !defined(XGBOOST_USE_CUDA)
std::tuple<double, double, double> GPUBinaryROCAUC(common::Span<float const>, MetaInfo const &,
                                                   std::int32_t,
                                                   std::shared_ptr<DeviceAUCCache> *) {
  common::AssertGPUSupport();
  return {};
}

double GPUMultiClassROCAUC(Context const *, common::Span<float const>, MetaInfo const &,
                           std::shared_ptr<DeviceAUCCache> *, std::size_t) {
  common::AssertGPUSupport();
  return 0.0;
}

std::pair<double, std::uint32_t> GPURankingAUC(Context const *, common::Span<float const>,
                                               MetaInfo const &,
                                               std::shared_ptr<DeviceAUCCache> *) {
  common::AssertGPUSupport();
  return {};
}
struct DeviceAUCCache {};
#endif  // !defined(XGBOOST_USE_CUDA)

class EvalPRAUC : public EvalAUC<EvalPRAUC> {
  std::shared_ptr<DeviceAUCCache> d_cache_;

 public:
  std::tuple<double, double, double>
  EvalBinary(HostDeviceVector<float> const &predts, MetaInfo const &info) {
    double pr, re, auc;
    if (ctx_->gpu_id == Context::kCpuId) {
      std::tie(pr, re, auc) =
          BinaryPRAUC(ctx_, predts.ConstHostSpan(), info.labels.HostView().Slice(linalg::All(), 0),
                      common::OptionalWeights{info.weights_.ConstHostSpan()});
    } else {
      std::tie(pr, re, auc) = GPUBinaryPRAUC(predts.ConstDeviceSpan(), info,
                                             ctx_->gpu_id, &this->d_cache_);
    }
    return std::make_tuple(pr, re, auc);
  }

  double EvalMultiClass(HostDeviceVector<float> const &predts, MetaInfo const &info,
                        size_t n_classes) {
    if (ctx_->gpu_id == Context::kCpuId) {
      auto n_threads = this->ctx_->Threads();
      return MultiClassOVR(ctx_, predts.ConstHostSpan(), info, n_classes, n_threads, BinaryPRAUC);
    } else {
      return GPUMultiClassPRAUC(ctx_, predts.ConstDeviceSpan(), info, &d_cache_, n_classes);
    }
  }

  std::pair<double, uint32_t> EvalRanking(HostDeviceVector<float> const &predts,
                                          MetaInfo const &info) {
    double auc{0};
    uint32_t valid_groups = 0;
    auto n_threads = ctx_->Threads();
    if (ctx_->gpu_id == Context::kCpuId) {
      auto labels = info.labels.Data()->ConstHostSpan();
      if (std::any_of(labels.cbegin(), labels.cend(), PRAUCLabelInvalid{})) {
        InvalidLabels();
      }
      std::tie(auc, valid_groups) =
          RankingAUC<false>(ctx_, predts.ConstHostVector(), info, n_threads);
    } else {
      std::tie(auc, valid_groups) =
          GPURankingPRAUC(ctx_, predts.ConstDeviceSpan(), info, &d_cache_);
    }
    return std::make_pair(auc, valid_groups);
  }

 public:
  const char *Name() const override { return "aucpr"; }
};

XGBOOST_REGISTER_METRIC(AUCPR, "aucpr")
    .describe("Area under PR curve for both classification and rank.")
    .set_body([](char const *) { return new EvalPRAUC{}; });

#if !defined(XGBOOST_USE_CUDA)
std::tuple<double, double, double> GPUBinaryPRAUC(common::Span<float const>, MetaInfo const &,
                                                  std::int32_t, std::shared_ptr<DeviceAUCCache> *) {
  common::AssertGPUSupport();
  return {};
}

double GPUMultiClassPRAUC(Context const *, common::Span<float const>, MetaInfo const &,
                          std::shared_ptr<DeviceAUCCache> *, std::size_t) {
  common::AssertGPUSupport();
  return {};
}

std::pair<double, std::uint32_t> GPURankingPRAUC(Context const *, common::Span<float const>,
                                                 MetaInfo const &,
                                                 std::shared_ptr<DeviceAUCCache> *) {
  common::AssertGPUSupport();
  return {};
}
#endif
}  // namespace metric
}  // namespace xgboost
