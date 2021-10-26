/*!
 * Copyright 2021 by XGBoost Contributors
 */
#include <array>
#include <atomic>
#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <tuple>
#include <vector>

#include "rabit/rabit.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/metric.h"

#include "auc.h"

#include "../common/common.h"
#include "../common/math.h"
#include "../common/threading_utils.h"

namespace xgboost {
namespace metric {
/**
 * Calculate AUC for binary classification problem.  This function does not normalize the
 * AUC by 1 / (num_positive * num_negative), instead it returns a tuple for caller to
 * handle the normalization.
 */
template <typename Fn>
std::tuple<float, float, float>
BinaryAUC(common::Span<float const> predts, common::Span<float const> labels,
          OptionalWeights weights,
          std::vector<size_t> const &sorted_idx, Fn &&area_fn) {
  CHECK(!labels.empty());
  CHECK_EQ(labels.size(), predts.size());
  auto p_predts = predts.data();
  auto p_labels = labels.data();

  float auc{0};

  float label = p_labels[sorted_idx.front()];
  float w = weights[sorted_idx[0]];
  float fp = (1.0 - label) * w, tp = label * w;
  float tp_prev = 0, fp_prev = 0;
  // TODO(jiaming): We can parallize this if we have a parallel scan for CPU.
  for (size_t i = 1; i < sorted_idx.size(); ++i) {
    if (p_predts[sorted_idx[i]] != p_predts[sorted_idx[i - 1]]) {
      auc += area_fn(fp_prev, fp, tp_prev, tp);
      tp_prev = tp;
      fp_prev = fp;
    }
    label = p_labels[sorted_idx[i]];
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
float MultiClassOVR(common::Span<float const> predts, MetaInfo const &info,
                    size_t n_classes, int32_t n_threads,
                    BinaryAUC &&binary_auc) {
  CHECK_NE(n_classes, 0);
  auto const &labels = info.labels_.ConstHostVector();

  std::vector<float> results(n_classes * 3, 0);
  auto s_results = common::Span<float>(results);
  auto local_area = s_results.subspan(0, n_classes);
  auto tp = s_results.subspan(n_classes, n_classes);
  auto auc = s_results.subspan(2 * n_classes, n_classes);
  auto weights = OptionalWeights{info.weights_.ConstHostSpan()};

  if (!info.labels_.Empty()) {
    common::ParallelFor(n_classes, n_threads, [&](auto c) {
      std::vector<float> proba(info.labels_.Size());
      std::vector<float> response(info.labels_.Size());
      for (size_t i = 0; i < proba.size(); ++i) {
        proba[i] = predts[i * n_classes + c];
        response[i] = labels[i] == c ? 1.0f : 0.0;
      }
      float fp;
      std::tie(fp, tp[c], auc[c]) = binary_auc(proba, response, weights);
      local_area[c] = fp * tp[c];
    });
  }

  // we have 2 averages going in here, first is among workers, second is among
  // classes. allreduce sums up fp/tp auc for each class.
  rabit::Allreduce<rabit::op::Sum>(results.data(), results.size());
  float auc_sum{0};
  float tp_sum{0};
  for (size_t c = 0; c < n_classes; ++c) {
    if (local_area[c] != 0) {
      // normalize and weight it by prevalence.  After allreduce, `local_area`
      // means the total covered area (not area under curve, rather it's the
      // accessible area for each worker) for each class.
      auc_sum += auc[c] / local_area[c] * tp[c];
      tp_sum += tp[c];
    } else {
      auc_sum = std::numeric_limits<float>::quiet_NaN();
      break;
    }
  }
  if (tp_sum == 0 || std::isnan(auc_sum)) {
    auc_sum = std::numeric_limits<float>::quiet_NaN();
  } else {
    auc_sum /= tp_sum;
  }
  return auc_sum;
}

std::tuple<float, float, float> BinaryROCAUC(common::Span<float const> predts,
                                             common::Span<float const> labels,
                                             OptionalWeights weights) {
  auto const sorted_idx = common::ArgSort<size_t>(predts, std::greater<>{});
  return BinaryAUC(predts, labels, weights, sorted_idx, TrapezoidArea);
}

/**
 * Calculate AUC for 1 ranking group;
 */
float GroupRankingROC(common::Span<float const> predts,
                      common::Span<float const> labels, float w) {
  // on ranking, we just count all pairs.
  float auc{0};
  auto const sorted_idx = common::ArgSort<size_t>(labels, std::greater<>{});
  w = common::Sqr(w);

  float sum_w = 0.0f;
  for (size_t i = 0; i < labels.size(); ++i) {
    for (size_t j = i + 1; j < labels.size(); ++j) {
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
std::tuple<float, float, float> BinaryPRAUC(common::Span<float const> predts,
                                            common::Span<float const> labels,
                                            OptionalWeights weights) {
  auto const sorted_idx = common::ArgSort<size_t>(predts, std::greater<>{});
  float total_pos{0}, total_neg{0};
  for (size_t i = 0; i < labels.size(); ++i) {
    auto w = weights[i];
    total_pos += w * labels[i];
    total_neg += w * (1.0f - labels[i]);
  }
  if (total_pos <= 0 || total_neg <= 0) {
    return {1.0f, 1.0f, std::numeric_limits<float>::quiet_NaN()};
  }
  auto fn = [total_pos](float fp_prev, float fp, float tp_prev, float tp) {
    return detail::CalcDeltaPRAUC(fp_prev, fp, tp_prev, tp, total_pos);
  };

  float tp{0}, fp{0}, auc{0};
  std::tie(fp, tp, auc) = BinaryAUC(predts, labels, weights, sorted_idx, fn);
  return std::make_tuple(1.0, 1.0, auc);
}


/**
 * Cast LTR problem to binary classification problem by comparing pairs.
 */
template <bool is_roc>
std::pair<float, uint32_t> RankingAUC(std::vector<float> const &predts,
                                      MetaInfo const &info, int32_t n_threads) {
  CHECK_GE(info.group_ptr_.size(), 2);
  uint32_t n_groups = info.group_ptr_.size() - 1;
  auto s_predts = common::Span<float const>{predts};
  auto s_labels = info.labels_.ConstHostSpan();
  auto s_weights = info.weights_.ConstHostSpan();

  std::atomic<uint32_t> invalid_groups{0};

  std::vector<double> auc_tloc(n_threads, 0);
  common::ParallelFor(n_groups, n_threads, [&](size_t g) {
    g += 1;  // indexing needs to start from 1
    size_t cnt = info.group_ptr_[g] - info.group_ptr_[g - 1];
    float w = s_weights.empty() ? 1.0f : s_weights[g - 1];
    auto g_predts = s_predts.subspan(info.group_ptr_[g - 1], cnt);
    auto g_labels = s_labels.subspan(info.group_ptr_[g - 1], cnt);
    float auc;
    if (is_roc && g_labels.size() < 3) {
      // With 2 documents, there's only 1 comparison can be made.  So either
      // TP or FP will be zero.
      invalid_groups++;
      auc = 0;
    } else {
      if (is_roc) {
        auc = GroupRankingROC(g_predts, g_labels, w);
      } else {
        auc = std::get<2>(BinaryPRAUC(g_predts, g_labels, OptionalWeights{w}));
      }
      if (std::isnan(auc)) {
        invalid_groups++;
        auc = 0;
      }
    }
    auc_tloc[omp_get_thread_num()] += auc;
  });
  float sum_auc = std::accumulate(auc_tloc.cbegin(), auc_tloc.cend(), 0.0);

  return std::make_pair(sum_auc, n_groups - invalid_groups);
}

template <typename Curve>
class EvalAUC : public Metric {
  float Eval(const HostDeviceVector<bst_float> &preds, const MetaInfo &info,
             bool distributed) override {
    float auc {0};
    if (tparam_->gpu_id != GenericParameter::kCpuId) {
      preds.SetDevice(tparam_->gpu_id);
      info.labels_.SetDevice(tparam_->gpu_id);
      info.weights_.SetDevice(tparam_->gpu_id);
    }
    //  We use the global size to handle empty dataset.
    std::array<size_t, 2> meta{info.labels_.Size(), preds.Size()};
    rabit::Allreduce<rabit::op::Max>(meta.data(), meta.size());
    if (meta[0] == 0) {
      // Empty across all workers, which is not supported.
      auc = std::numeric_limits<float>::quiet_NaN();
    } else if (!info.group_ptr_.empty()) {
      /**
       * learning to rank
       */
      if (!info.weights_.Empty()) {
        CHECK_EQ(info.weights_.Size(), info.group_ptr_.size() - 1);
      }
      uint32_t valid_groups = 0;
      if (!info.labels_.Empty()) {
        CHECK_EQ(info.group_ptr_.back(), info.labels_.Size());
        std::tie(auc, valid_groups) =
            static_cast<Curve *>(this)->EvalRanking(preds, info);
      }
      if (valid_groups != info.group_ptr_.size() - 1) {
        InvalidGroupAUC();
      }

      std::array<float, 2> results{auc, static_cast<float>(valid_groups)};
      rabit::Allreduce<rabit::op::Sum>(results.data(), results.size());
      auc = results[0];
      valid_groups = static_cast<uint32_t>(results[1]);

      if (valid_groups <= 0) {
        auc = std::numeric_limits<float>::quiet_NaN();
      } else {
        auc /= valid_groups;
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
      float fp{0}, tp{0};
      if (!(preds.Empty() || info.labels_.Empty())) {
        std::tie(fp, tp, auc) =
            static_cast<Curve *>(this)->EvalBinary(preds, info);
      }
      float local_area = fp * tp;
      std::array<float, 2> result{auc, local_area};
      rabit::Allreduce<rabit::op::Sum>(result.data(), result.size());
      std::tie(auc, local_area) = common::UnpackArr(std::move(result));
      if (local_area <= 0) {
        // the dataset across all workers have only positive or negative sample
        auc = std::numeric_limits<float>::quiet_NaN();
      } else {
        CHECK_LE(auc, local_area);
        // normalization
        auc = auc / local_area;
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
  std::pair<float, uint32_t> EvalRanking(HostDeviceVector<float> const &predts,
                                         MetaInfo const &info) {
    float auc{0};
    uint32_t valid_groups = 0;
    auto n_threads = tparam_->Threads();
    if (tparam_->gpu_id == GenericParameter::kCpuId) {
      std::tie(auc, valid_groups) =
          RankingAUC<true>(predts.ConstHostVector(), info, n_threads);
    } else {
      std::tie(auc, valid_groups) = GPURankingAUC(
          predts.ConstDeviceSpan(), info, tparam_->gpu_id, &this->d_cache_);
    }
    return std::make_pair(auc, valid_groups);
  }

  float EvalMultiClass(HostDeviceVector<float> const &predts,
                       MetaInfo const &info, size_t n_classes) {
    float auc{0};
    auto n_threads = tparam_->Threads();
    CHECK_NE(n_classes, 0);
    if (tparam_->gpu_id == GenericParameter::kCpuId) {
      auc = MultiClassOVR(predts.ConstHostVector(), info, n_classes, n_threads,
                          BinaryROCAUC);
    } else {
      auc = GPUMultiClassROCAUC(predts.ConstDeviceSpan(), info, tparam_->gpu_id,
                                &this->d_cache_, n_classes);
    }
    return auc;
  }

  std::tuple<float, float, float>
  EvalBinary(HostDeviceVector<float> const &predts, MetaInfo const &info) {
    float fp, tp, auc;
    if (tparam_->gpu_id == GenericParameter::kCpuId) {
      std::tie(fp, tp, auc) =
          BinaryROCAUC(predts.ConstHostVector(), info.labels_.ConstHostVector(),
                       OptionalWeights{info.weights_.ConstHostSpan()});
    } else {
      std::tie(fp, tp, auc) = GPUBinaryROCAUC(predts.ConstDeviceSpan(), info,
                                              tparam_->gpu_id, &this->d_cache_);
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
std::tuple<float, float, float>
GPUBinaryROCAUC(common::Span<float const> predts, MetaInfo const &info,
                int32_t device, std::shared_ptr<DeviceAUCCache> *p_cache) {
  common::AssertGPUSupport();
  return std::make_tuple(0.0f, 0.0f, 0.0f);
}

float GPUMultiClassROCAUC(common::Span<float const> predts,
                          MetaInfo const &info, int32_t device,
                          std::shared_ptr<DeviceAUCCache> *cache,
                          size_t n_classes) {
  common::AssertGPUSupport();
  return 0;
}

std::pair<float, uint32_t>
GPURankingAUC(common::Span<float const> predts, MetaInfo const &info,
              int32_t device, std::shared_ptr<DeviceAUCCache> *p_cache) {
  common::AssertGPUSupport();
  return std::make_pair(0.0f, 0u);
}
struct DeviceAUCCache {};
#endif  // !defined(XGBOOST_USE_CUDA)

class EvalAUCPR : public EvalAUC<EvalAUCPR> {
  std::shared_ptr<DeviceAUCCache> d_cache_;

 public:
  std::tuple<float, float, float>
  EvalBinary(HostDeviceVector<float> const &predts, MetaInfo const &info) {
    float pr, re, auc;
    if (tparam_->gpu_id == GenericParameter::kCpuId) {
      std::tie(pr, re, auc) =
          BinaryPRAUC(predts.ConstHostSpan(), info.labels_.ConstHostSpan(),
                      OptionalWeights{info.weights_.ConstHostSpan()});
    } else {
      std::tie(pr, re, auc) = GPUBinaryPRAUC(predts.ConstDeviceSpan(), info,
                                             tparam_->gpu_id, &this->d_cache_);
    }
    return std::make_tuple(pr, re, auc);
  }

  float EvalMultiClass(HostDeviceVector<float> const &predts,
                       MetaInfo const &info, size_t n_classes) {
    if (tparam_->gpu_id == GenericParameter::kCpuId) {
      auto n_threads = this->tparam_->Threads();
      return MultiClassOVR(predts.ConstHostSpan(), info, n_classes, n_threads,
                           BinaryPRAUC);
    } else {
      return GPUMultiClassPRAUC(predts.ConstDeviceSpan(), info, tparam_->gpu_id,
                                &d_cache_, n_classes);
    }
  }

  std::pair<float, uint32_t> EvalRanking(HostDeviceVector<float> const &predts,
                                         MetaInfo const &info) {
    float auc{0};
    uint32_t valid_groups = 0;
    auto n_threads = tparam_->Threads();
    if (tparam_->gpu_id == GenericParameter::kCpuId) {
      auto labels = info.labels_.ConstHostSpan();
      if (std::any_of(labels.cbegin(), labels.cend(), PRAUCLabelInvalid{})) {
        InvalidLabels();
      }
      std::tie(auc, valid_groups) =
          RankingAUC<false>(predts.ConstHostVector(), info, n_threads);
    } else {
      std::tie(auc, valid_groups) = GPURankingPRAUC(
          predts.ConstDeviceSpan(), info, tparam_->gpu_id, &d_cache_);
    }
    return std::make_pair(auc, valid_groups);
  }

 public:
  const char *Name() const override { return "aucpr"; }
};

XGBOOST_REGISTER_METRIC(AUCPR, "aucpr")
    .describe("Area under PR curve for both classification and rank.")
    .set_body([](char const *) { return new EvalAUCPR{}; });

#if !defined(XGBOOST_USE_CUDA)
std::tuple<float, float, float>
GPUBinaryPRAUC(common::Span<float const> predts, MetaInfo const &info,
               int32_t device, std::shared_ptr<DeviceAUCCache> *p_cache) {
  common::AssertGPUSupport();
  return {};
}

float GPUMultiClassPRAUC(common::Span<float const> predts, MetaInfo const &info,
                         int32_t device, std::shared_ptr<DeviceAUCCache> *cache,
                         size_t n_classes) {
  common::AssertGPUSupport();
  return {};
}

std::pair<float, uint32_t>
GPURankingPRAUC(common::Span<float const> predts, MetaInfo const &info,
                int32_t device, std::shared_ptr<DeviceAUCCache> *cache) {
  common::AssertGPUSupport();
  return {};
}
#endif
}  // namespace metric
}  // namespace xgboost
