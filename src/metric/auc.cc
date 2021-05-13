/*!
 * Copyright 2021 by XGBoost Contributors
 */
#include <array>
#include <atomic>
#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <tuple>
#include <vector>

#include "rabit/rabit.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/metric.h"
#include "auc.h"
#include "../common/common.h"
#include "../common/math.h"

namespace xgboost {
namespace metric {

namespace detail {
template <class T, std::size_t N, std::size_t... Idx>
constexpr auto UnpackArr(std::array<T, N> &&arr, std::index_sequence<Idx...>) {
  return std::make_tuple(std::forward<std::array<T, N>>(arr)[Idx]...);
}
}  // namespace detail

template <class T, std::size_t N>
constexpr auto UnpackArr(std::array<T, N> &&arr) {
  return detail::UnpackArr(std::forward<std::array<T, N>>(arr),
                           std::make_index_sequence<N>{});
}

/**
 * Calculate AUC for binary classification problem.  This function does not normalize the
 * AUC by 1 / (num_positive * num_negative), instead it returns a tuple for caller to
 * handle the normalization.
 */
std::tuple<float, float, float> BinaryAUC(std::vector<float> const &predts,
                                          std::vector<float> const &labels,
                                          std::vector<float> const &weights) {
  CHECK(!labels.empty());
  CHECK_EQ(labels.size(), predts.size());

  float auc {0};
  auto const sorted_idx = common::ArgSort<size_t>(
      common::Span<float const>(predts), std::greater<>{});

  auto get_weight = [&](size_t i) {
    return weights.empty() ? 1.0f : weights[sorted_idx[i]];
  };
  float label = labels[sorted_idx.front()];
  float w = get_weight(0);
  float fp = (1.0 - label) * w, tp = label * w;
  float tp_prev = 0, fp_prev = 0;
  // TODO(jiaming): We can parallize this if we have a parallel scan for CPU.
  for (size_t i = 1; i < sorted_idx.size(); ++i) {
    if (predts[sorted_idx[i]] != predts[sorted_idx[i-1]]) {
      auc += TrapesoidArea(fp_prev, fp, tp_prev, tp);
      tp_prev = tp;
      fp_prev = fp;
    }
    label = labels[sorted_idx[i]];
    float w = get_weight(i);
    fp += (1.0f - label) * w;
    tp += label * w;
  }

  auc += TrapesoidArea(fp_prev, fp, tp_prev, tp);
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
float MultiClassOVR(std::vector<float> const& predts, MetaInfo const& info, size_t n_classes) {
  CHECK_NE(n_classes, 0);
  auto const& labels = info.labels_.ConstHostVector();

  std::vector<float> results(n_classes * 3, 0);
  auto s_results = common::Span<float>(results);
  auto local_area = s_results.subspan(0, n_classes);
  auto tp = s_results.subspan(n_classes, n_classes);
  auto auc = s_results.subspan(2 * n_classes, n_classes);

  if (!info.labels_.Empty()) {
    dmlc::OMPException omp_handler;
#pragma omp parallel for
    for (omp_ulong c = 0; c < n_classes; ++c) {
      omp_handler.Run([&]() {
        std::vector<float> proba(info.labels_.Size());
        std::vector<float> response(info.labels_.Size());
        for (size_t i = 0; i < proba.size(); ++i) {
          proba[i] = predts[i * n_classes + c];
          response[i] = labels[i] == c ? 1.0f : 0.0;
        }
        float fp;
        std::tie(fp, tp[c], auc[c]) =
            BinaryAUC(proba, response, info.weights_.ConstHostVector());
        local_area[c] = fp * tp[c];
      });
    }
    omp_handler.Rethrow();
  }

  // we have 2 averages going in here, first is among workers, second is among classes.
  // allreduce sums up fp/tp auc for each class.
  rabit::Allreduce<rabit::op::Sum>(results.data(), results.size());
  float auc_sum{0};
  float tp_sum{0};
  for (size_t c = 0; c < n_classes; ++c) {
    if (local_area[c] != 0) {
      // normalize and weight it by prevalence.  After allreduce, `local_area` means the
      // total covered area (not area under curve, rather it's the accessible are for each
      // worker) for each class.
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

/**
 * Calculate AUC for 1 ranking group;
 */
float GroupRankingAUC(common::Span<float const> predts,
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
 * Cast LTR problem to binary classification problem by comparing pairs.
 */
std::pair<float, uint32_t> RankingAUC(std::vector<float> const &predts,
                                      MetaInfo const &info) {
  CHECK_GE(info.group_ptr_.size(), 2);
  uint32_t n_groups = info.group_ptr_.size() - 1;
  float sum_auc = 0;
  auto s_predts = common::Span<float const>{predts};
  auto s_labels = info.labels_.ConstHostSpan();
  auto s_weights = info.weights_.ConstHostSpan();

  std::atomic<uint32_t> invalid_groups{0};
  dmlc::OMPException omp_handler;

#pragma omp parallel for reduction(+:sum_auc)
  for (omp_ulong g = 1; g < info.group_ptr_.size(); ++g) {
    omp_handler.Run([&]() {
      size_t cnt = info.group_ptr_[g] - info.group_ptr_[g - 1];
      float w = s_weights.empty() ? 1.0f : s_weights[g - 1];
      auto g_predts = s_predts.subspan(info.group_ptr_[g - 1], cnt);
      auto g_labels = s_labels.subspan(info.group_ptr_[g - 1], cnt);
      float auc;
      if (g_labels.size() < 3) {
        // With 2 documents, there's only 1 comparison can be made.  So either
        // TP or FP will be zero.
        invalid_groups++;
        auc = 0;
      } else {
        auc = GroupRankingAUC(g_predts, g_labels, w);
      }
      sum_auc += auc;
    });
  }
  omp_handler.Rethrow();

  if (invalid_groups != 0) {
    InvalidGroupAUC();
  }

  return std::make_pair(sum_auc, n_groups - invalid_groups);
}

class EvalAUC : public Metric {
  std::shared_ptr<DeviceAUCCache> d_cache_;

 public:
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

    if (!info.group_ptr_.empty()) {
      /**
       * learning to rank
       */
      if (!info.weights_.Empty()) {
        CHECK_EQ(info.weights_.Size(), info.group_ptr_.size() - 1);
      }
      uint32_t valid_groups = 0;
      if (!info.labels_.Empty()) {
        CHECK_EQ(info.group_ptr_.back(), info.labels_.Size());
        if (tparam_->gpu_id == GenericParameter::kCpuId) {
          std::tie(auc, valid_groups) =
              RankingAUC(preds.ConstHostVector(), info);
        } else {
          std::tie(auc, valid_groups) = GPURankingAUC(
              preds.ConstDeviceSpan(), info, tparam_->gpu_id, &this->d_cache_);
        }
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
      if (tparam_->gpu_id == GenericParameter::kCpuId) {
        auc = MultiClassOVR(preds.ConstHostVector(), info, n_classes);
      } else {
        auc = GPUMultiClassAUCOVR(preds.ConstDeviceSpan(), info, tparam_->gpu_id,
                                  &this->d_cache_, n_classes);
      }
    } else {
      /**
       * binary classification
       */
      float fp{0}, tp{0};
      if (!(preds.Empty() || info.labels_.Empty())) {
        if (tparam_->gpu_id == GenericParameter::kCpuId) {
          std::tie(fp, tp, auc) =
              BinaryAUC(preds.ConstHostVector(), info.labels_.ConstHostVector(),
                        info.weights_.ConstHostVector());
        } else {
          std::tie(fp, tp, auc) = GPUBinaryAUC(
              preds.ConstDeviceSpan(), info, tparam_->gpu_id, &this->d_cache_);
        }
      }
      float local_area = fp * tp;
      std::array<float, 2> result{auc, local_area};
      rabit::Allreduce<rabit::op::Sum>(result.data(), result.size());
      std::tie(auc, local_area) = UnpackArr(std::move(result));
      if (local_area <= 0) {
        // the dataset across all workers have only positive or negative sample
        auc = std::numeric_limits<float>::quiet_NaN();
      } else {
        // normalization
        auc = auc / local_area;
      }
    }
    if (std::isnan(auc)) {
      LOG(WARNING) << "Dataset contains only positive or negative samples.";
    }
    return auc;
  }

  char const* Name() const override {
    return "auc";
  }
};

XGBOOST_REGISTER_METRIC(EvalBinaryAUC, "auc")
.describe("Receiver Operating Characteristic Area Under the Curve.")
.set_body([](const char*) { return new EvalAUC(); });

#if !defined(XGBOOST_USE_CUDA)
std::tuple<float, float, float>
GPUBinaryAUC(common::Span<float const> predts, MetaInfo const &info,
             int32_t device, std::shared_ptr<DeviceAUCCache> *p_cache) {
  common::AssertGPUSupport();
  return std::make_tuple(0.0f, 0.0f, 0.0f);
}

float GPUMultiClassAUCOVR(common::Span<float const> predts, MetaInfo const &info,
                          int32_t device, std::shared_ptr<DeviceAUCCache>* cache,
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
}  // namespace metric
}  // namespace xgboost
