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

#include "./math.h"                      // for CloseTo
#include "dmlc/parameter.h"              // for FieldEntry, DMLC_DECLARE_FIELD
#include "error_msg.h"                   // for GroupWeight, GroupSize
#include "xgboost/base.h"                // for XGBOOST_DEVICE, bst_group_t
#include "xgboost/context.h"             // for Context
#include "xgboost/data.h"                // for MetaInfo
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/linalg.h"              // for Vector, VectorView, Tensor
#include "xgboost/logging.h"             // for LogCheck_EQ, CHECK_EQ, CHECK
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
  PairMethod lambdarank_pair_method{PairMethod::kMean};  // NOLINT
  std::size_t lambdarank_num_pair_per_sample{NotSet()};  // NOLINT

 public:
  static constexpr position_t NotSet() { return std::numeric_limits<position_t>::max(); }

  // unbiased
  bool lambdarank_unbiased{false};
  double lambdarank_bias_norm{2.0};
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
        .set_default(PairMethod::kMean)
        .add_enum("mean", PairMethod::kMean)
        .add_enum("topk", PairMethod::kTopK)
        .describe("Method for constructing pairs.");
    DMLC_DECLARE_FIELD(lambdarank_num_pair_per_sample)
        .set_default(NotSet())
        .set_lower_bound(1)
        .describe("Number of pairs for each sample in the list.");
    DMLC_DECLARE_FIELD(lambdarank_unbiased)
        .set_default(false)
        .describe("Unbiased lambda mart. Use IPW to debias click position");
    DMLC_DECLARE_FIELD(lambdarank_bias_norm)
        .set_default(2.0)
        .set_lower_bound(0.0)
        .describe("Lp regularization for unbiased lambdarank.");
    DMLC_DECLARE_FIELD(ndcg_exp_gain)
        .set_default(true)
        .describe("When set to true, the label gain is 2^rel - 1, otherwise it's rel.");
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
