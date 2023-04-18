/**
 * Copyright 2023 by XGBoost contributors
 */
#ifndef XGBOOST_COMMON_RANKING_UTILS_H_
#define XGBOOST_COMMON_RANKING_UTILS_H_
#include <algorithm>                     // for min
#include <cmath>                         // for log2, fabs, floor
#include <cstddef>                       // for size_t
#include <cstdint>                       // for uint32_t, uint8_t, int32_t
#include <limits>                        // for numeric_limits
#include <string>                        // for char_traits, string
#include <vector>                        // for vector

#include "dmlc/parameter.h"              // for FieldEntry, DMLC_DECLARE_FIELD
#include "error_msg.h"                   // for GroupWeight, GroupSize
#include "xgboost/base.h"                // for XGBOOST_DEVICE, bst_group_t
#include "xgboost/context.h"             // for Context
#include "xgboost/data.h"                // for MetaInfo
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/linalg.h"              // for Vector, VectorView, Tensor
#include "xgboost/logging.h"             // for CHECK_EQ, CHECK
#include "xgboost/parameter.h"           // for XGBoostParameter
#include "xgboost/span.h"                // for Span
#include "xgboost/string_view.h"         // for StringView

namespace xgboost::ltr {
/**
 * \brief Relevance degree
 */
using rel_degree_t = std::uint32_t;  // NOLINT
/**
 * \brief top-k position
 */
using position_t = std::uint32_t;  // NOLINT

/**
 * \brief Maximum relevance degree for NDCG
 */
constexpr std::size_t MaxRel() { return sizeof(rel_degree_t) * 8 - 1; }
static_assert(MaxRel() == 31);

XGBOOST_DEVICE inline double CalcDCGGain(rel_degree_t label) {
  return static_cast<double>((1u << label) - 1);
}

XGBOOST_DEVICE inline double CalcDCGDiscount(std::size_t idx) {
  return 1.0 / std::log2(static_cast<double>(idx) + 2.0);
}

XGBOOST_DEVICE inline double CalcInvIDCG(double idcg) {
  auto inv_idcg = (idcg == 0.0 ? 0.0 : (1.0 / idcg));  // handle irrelevant document
  return inv_idcg;
}

enum class PairMethod : std::int32_t {
  kTopK = 0,
  kMean = 1,
};
}  // namespace xgboost::ltr

DECLARE_FIELD_ENUM_CLASS(xgboost::ltr::PairMethod);

namespace xgboost::ltr {
struct LambdaRankParam : public XGBoostParameter<LambdaRankParam> {
 private:
  static constexpr position_t DefaultK() { return 32; }
  static constexpr position_t DefaultSamplePairs() { return 1; }

 protected:
  // pairs
  // should be accessed by getter for auto configuration.
  // nolint so that we can keep the string name.
  PairMethod lambdarank_pair_method{PairMethod::kTopK};  // NOLINT
  std::size_t lambdarank_num_pair_per_sample{NotSet()};  // NOLINT

 public:
  static constexpr position_t NotSet() { return std::numeric_limits<position_t>::max(); }

  // unbiased
  bool lambdarank_unbiased{false};
  double lambdarank_bias_norm{1.0};
  // ndcg
  bool ndcg_exp_gain{true};

  bool operator==(LambdaRankParam const& that) const {
    return lambdarank_pair_method == that.lambdarank_pair_method &&
           lambdarank_num_pair_per_sample == that.lambdarank_num_pair_per_sample &&
           lambdarank_unbiased == that.lambdarank_unbiased &&
           lambdarank_bias_norm == that.lambdarank_bias_norm && ndcg_exp_gain == that.ndcg_exp_gain;
  }
  bool operator!=(LambdaRankParam const& that) const { return !(*this == that); }

  [[nodiscard]] double Regularizer() const { return 1.0 / (1.0 + this->lambdarank_bias_norm); }

  /**
   * \brief Get number of pairs for each sample
   */
  [[nodiscard]] position_t NumPair() const {
    if (lambdarank_num_pair_per_sample == NotSet()) {
      switch (lambdarank_pair_method) {
        case PairMethod::kMean:
          return DefaultSamplePairs();
        case PairMethod::kTopK:
          return DefaultK();
      }
    } else {
      return lambdarank_num_pair_per_sample;
    }
    LOG(FATAL) << "Unreachable.";
    return 0;
  }

  [[nodiscard]] bool HasTruncation() const { return lambdarank_pair_method == PairMethod::kTopK; }

  // Used for evaluation metric and cache initialization, iterate through top-k or the whole list
  [[nodiscard]] auto TopK() const {
    if (HasTruncation()) {
      return NumPair();
    } else {
      return NotSet();
    }
  }

  DMLC_DECLARE_PARAMETER(LambdaRankParam) {
    DMLC_DECLARE_FIELD(lambdarank_pair_method)
        .set_default(PairMethod::kTopK)
        .add_enum("mean", PairMethod::kMean)
        .add_enum("topk", PairMethod::kTopK)
        .describe("Method for constructing pairs.");
    DMLC_DECLARE_FIELD(lambdarank_num_pair_per_sample)
        .set_default(NotSet())
        .set_lower_bound(1)
        .describe("Number of pairs for each sample in the list.");
    DMLC_DECLARE_FIELD(lambdarank_unbiased)
        .set_default(false)
        .describe("Unbiased lambda mart. Use extended IPW to debias click position");
    DMLC_DECLARE_FIELD(lambdarank_bias_norm)
        .set_default(1.0)
        .set_lower_bound(0.0)
        .describe("Lp regularization for unbiased lambdarank.");
    DMLC_DECLARE_FIELD(ndcg_exp_gain)
        .set_default(true)
        .describe("When set to true, the label gain is 2^rel - 1, otherwise it's rel.");
  }
};

/**
 * \brief Common cached items for ranking tasks.
 */
class RankingCache {
 private:
  void InitOnCPU(Context const* ctx, MetaInfo const& info);
  void InitOnCUDA(Context const* ctx, MetaInfo const& info);
  // Cached parameter
  LambdaRankParam param_;
  // offset to data groups.
  HostDeviceVector<bst_group_t> group_ptr_;
  // store the sorted index of prediction.
  HostDeviceVector<std::size_t> sorted_idx_cache_;
  // Maximum size of group
  std::size_t max_group_size_{0};
  // Normalization for weight
  double weight_norm_{1.0};
  /**
   * CUDA cache
   */
  // offset to threads assigned to each group for gradient calculation
  HostDeviceVector<std::size_t> threads_group_ptr_;
  // Sorted index of label for finding buckets.
  HostDeviceVector<std::size_t> y_sorted_idx_cache_;
  // Cached labels sorted by the model
  HostDeviceVector<float> y_ranked_by_model_;
  // store rounding factor for objective for each group
  linalg::Vector<GradientPair> roundings_;
  // rounding factor for cost
  HostDeviceVector<double> cost_rounding_;
  // temporary storage for creating rounding factors. Stored as byte to avoid having cuda
  // data structure in here.
  HostDeviceVector<std::uint8_t> max_lambdas_;
  // total number of cuda threads used for gradient calculation
  std::size_t n_cuda_threads_{0};

  // Create model rank list on GPU
  common::Span<std::size_t const> MakeRankOnCUDA(Context const* ctx,
                                                 common::Span<float const> predt);
  // Create model rank list on CPU
  common::Span<std::size_t const> MakeRankOnCPU(Context const* ctx,
                                                common::Span<float const> predt);

 protected:
  [[nodiscard]] std::size_t MaxGroupSize() const { return max_group_size_; }

 public:
  RankingCache(Context const* ctx, MetaInfo const& info, LambdaRankParam const& p) : param_{p} {
    CHECK(param_.GetInitialised());
    if (!info.group_ptr_.empty()) {
      CHECK_EQ(info.group_ptr_.back(), info.labels.Size())
          << error::GroupSize() << "the size of label.";
    }
    if (ctx->IsCPU()) {
      this->InitOnCPU(ctx, info);
    } else {
      this->InitOnCUDA(ctx, info);
    }
    if (!info.weights_.Empty()) {
      CHECK_EQ(Groups(), info.weights_.Size()) << error::GroupWeight();
    }
  }
  [[nodiscard]] std::size_t MaxPositionSize() const {
    // Use truncation level as bound.
    if (param_.HasTruncation()) {
      return param_.NumPair();
    }
    // Hardcoded maximum size of positions to track. We don't need too many of them as the
    // bias decreases exponentially.
    return std::min(max_group_size_, static_cast<std::size_t>(32));
  }
  // Constructed as [1, n_samples] if group ptr is not supplied by the user
  common::Span<bst_group_t const> DataGroupPtr(Context const* ctx) const {
    group_ptr_.SetDevice(ctx->gpu_id);
    return ctx->IsCPU() ? group_ptr_.ConstHostSpan() : group_ptr_.ConstDeviceSpan();
  }

  [[nodiscard]] auto const& Param() const { return param_; }
  [[nodiscard]] std::size_t Groups() const { return group_ptr_.Size() - 1; }
  [[nodiscard]] double WeightNorm() const { return weight_norm_; }

  // Create a rank list by model prediction
  common::Span<std::size_t const> SortedIdx(Context const* ctx, common::Span<float const> predt) {
    if (sorted_idx_cache_.Empty()) {
      sorted_idx_cache_.SetDevice(ctx->gpu_id);
      sorted_idx_cache_.Resize(predt.size());
    }
    if (ctx->IsCPU()) {
      return this->MakeRankOnCPU(ctx, predt);
    } else {
      return this->MakeRankOnCUDA(ctx, predt);
    }
  }
  // The function simply returns a uninitialized buffer as this is only used by the
  // objective for creating pairs.
  common::Span<std::size_t> SortedIdxY(Context const* ctx, std::size_t n_samples) {
    CHECK(ctx->IsCUDA());
    if (y_sorted_idx_cache_.Empty()) {
      y_sorted_idx_cache_.SetDevice(ctx->gpu_id);
      y_sorted_idx_cache_.Resize(n_samples);
    }
    return y_sorted_idx_cache_.DeviceSpan();
  }
  common::Span<float> RankedY(Context const* ctx, std::size_t n_samples) {
    CHECK(ctx->IsCUDA());
    if (y_ranked_by_model_.Empty()) {
      y_ranked_by_model_.SetDevice(ctx->gpu_id);
      y_ranked_by_model_.Resize(n_samples);
    }
    return y_ranked_by_model_.DeviceSpan();
  }

  // CUDA cache getters, the cache is shared between metric and objective, some of these
  // fields are lazy initialized to avoid unnecessary allocation.
  [[nodiscard]] common::Span<std::size_t const> CUDAThreadsGroupPtr() const {
    CHECK(!threads_group_ptr_.Empty());
    return threads_group_ptr_.ConstDeviceSpan();
  }
  [[nodiscard]] std::size_t CUDAThreads() const { return n_cuda_threads_; }

  linalg::VectorView<GradientPair> CUDARounding(Context const* ctx) {
    if (roundings_.Size() == 0) {
      roundings_.SetDevice(ctx->gpu_id);
      roundings_.Reshape(Groups());
    }
    return roundings_.View(ctx->gpu_id);
  }
  common::Span<double> CUDACostRounding(Context const* ctx) {
    if (cost_rounding_.Size() == 0) {
      cost_rounding_.SetDevice(ctx->gpu_id);
      cost_rounding_.Resize(1);
    }
    return cost_rounding_.DeviceSpan();
  }
  template <typename Type>
  common::Span<Type> MaxLambdas(Context const* ctx, std::size_t n) {
    max_lambdas_.SetDevice(ctx->gpu_id);
    std::size_t bytes = n * sizeof(Type);
    if (bytes != max_lambdas_.Size()) {
      max_lambdas_.Resize(bytes);
    }
    return common::Span<Type>{reinterpret_cast<Type*>(max_lambdas_.DevicePointer()), n};
  }
};

class NDCGCache : public RankingCache {
  // NDCG discount
  HostDeviceVector<double> discounts_;
  // 1.0 / IDCG
  linalg::Vector<double> inv_idcg_;
  /**
   * CUDA cache
   */
  // store the intermediate DCG calculation result for metric
  linalg::Vector<double> dcg_;

 public:
  void InitOnCPU(Context const* ctx, MetaInfo const& info);
  void InitOnCUDA(Context const* ctx, MetaInfo const& info);

 public:
  NDCGCache(Context const* ctx, MetaInfo const& info, LambdaRankParam const& p)
      : RankingCache{ctx, info, p} {
    if (ctx->IsCPU()) {
      this->InitOnCPU(ctx, info);
    } else {
      this->InitOnCUDA(ctx, info);
    }
  }

  linalg::VectorView<double const> InvIDCG(Context const* ctx) const {
    return inv_idcg_.View(ctx->gpu_id);
  }
  common::Span<double const> Discount(Context const* ctx) const {
    return ctx->IsCPU() ? discounts_.ConstHostSpan() : discounts_.ConstDeviceSpan();
  }
  linalg::VectorView<double> Dcg(Context const* ctx) {
    if (dcg_.Size() == 0) {
      dcg_.SetDevice(ctx->gpu_id);
      dcg_.Reshape(this->Groups());
    }
    return dcg_.View(ctx->gpu_id);
  }
};

/**
 * \brief Validate label for NDCG
 *
 * \tparam NoneOf Implementation of std::none_of. Specified as a parameter to reuse the
 *                check for both CPU and GPU.
 */
template <typename NoneOf>
void CheckNDCGLabels(ltr::LambdaRankParam const& p, linalg::VectorView<float const> labels,
                     NoneOf none_of) {
  auto d_labels = labels.Values();
  if (p.ndcg_exp_gain) {
    auto label_is_integer =
        none_of(d_labels.data(), d_labels.data() + d_labels.size(), [] XGBOOST_DEVICE(float v) {
          auto l = std::floor(v);
          return std::fabs(l - v) > kRtEps || v < 0.0f;
        });
    CHECK(label_is_integer)
        << "When using relevance degree as target, label must be either 0 or positive integer.";
  }

  if (p.ndcg_exp_gain) {
    auto label_is_valid = none_of(d_labels.data(), d_labels.data() + d_labels.size(),
                                  [] XGBOOST_DEVICE(ltr::rel_degree_t v) { return v > MaxRel(); });
    CHECK(label_is_valid) << "Relevance degress must be lesser than or equal to " << MaxRel()
                          << " when the exponential NDCG gain function is used. "
                          << "Set `ndcg_exp_gain` to false to use custom DCG gain.";
  }
}

template <typename AllOf>
bool IsBinaryRel(linalg::VectorView<float const> label, AllOf all_of) {
  auto s_label = label.Values();
  return all_of(s_label.data(), s_label.data() + s_label.size(), [] XGBOOST_DEVICE(float y) {
    return std::abs(y - 1.0f) < kRtEps || std::abs(y - 0.0f) < kRtEps;
  });
}
/**
 * \brief Validate label for MAP
 *
 * \tparam Implementation of std::all_of. Specified as a parameter to reuse the check for
 *         both CPU and GPU.
 */
template <typename AllOf>
void CheckMapLabels(linalg::VectorView<float const> label, AllOf all_of) {
  auto s_label = label.Values();
  auto is_binary = IsBinaryRel(label, all_of);
  CHECK(is_binary) << "MAP can only be used with binary labels.";
}

class MAPCache : public RankingCache {
  // Total number of relevant documents for each group
  HostDeviceVector<double> n_rel_;
  // \sum l_k/k
  HostDeviceVector<double> acc_;
  HostDeviceVector<double> map_;
  // Number of samples in this dataset.
  std::size_t n_samples_{0};

  void InitOnCPU(Context const* ctx, MetaInfo const& info);
  void InitOnCUDA(Context const* ctx, MetaInfo const& info);

 public:
  MAPCache(Context const* ctx, MetaInfo const& info, LambdaRankParam const& p)
      : RankingCache{ctx, info, p}, n_samples_{static_cast<std::size_t>(info.num_row_)} {
    if (ctx->IsCPU()) {
      this->InitOnCPU(ctx, info);
    } else {
      this->InitOnCUDA(ctx, info);
    }
  }

  common::Span<double> NumRelevant(Context const* ctx) {
    if (n_rel_.Empty()) {
      n_rel_.SetDevice(ctx->gpu_id);
      n_rel_.Resize(n_samples_);
    }
    return ctx->IsCPU() ? n_rel_.HostSpan() : n_rel_.DeviceSpan();
  }
  common::Span<double> Acc(Context const* ctx) {
    if (acc_.Empty()) {
      acc_.SetDevice(ctx->gpu_id);
      acc_.Resize(n_samples_);
    }
    return ctx->IsCPU() ? acc_.HostSpan() : acc_.DeviceSpan();
  }
  common::Span<double> Map(Context const* ctx) {
    if (map_.Empty()) {
      map_.SetDevice(ctx->gpu_id);
      map_.Resize(this->Groups());
    }
    return ctx->IsCPU() ? map_.HostSpan() : map_.DeviceSpan();
  }
};

/**
 * \brief Parse name for ranking metric given parameters.
 *
 * \param [in] name   Null terminated string for metric name
 * \param [in] param  Null terminated string for parameter like the `3-` in `ndcg@3-`.
 * \param [out] topn  Top n documents parsed from param. Unchanged if it's not specified.
 * \param [out] minus Whether we should turn the score into loss. Unchanged if it's not
 *                    specified.
 *
 * \return The name of the metric.
 */
std::string ParseMetricName(StringView name, StringView param, position_t* topn, bool* minus);

/**
 * \brief Parse name for ranking metric given parameters.
 */
std::string MakeMetricName(StringView name, position_t topn, bool minus);
}  // namespace xgboost::ltr
#endif  // XGBOOST_COMMON_RANKING_UTILS_H_
