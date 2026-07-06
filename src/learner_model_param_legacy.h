/**
 * SPDX-FileCopyrightText: Copyright (c) 2014-2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <dmlc/parameter.h>

#include <algorithm>  // for find_if, max, none_of
#include <cstdint>    // for int32_t
#include <map>        // for map
#include <string>     // for string
#include <vector>     // for vector

#include "common/param_array.h"  // for ParamArray
#include "common/version.h"      // for Version
#include "xgboost/base.h"        // for Args, bst_feature_t, bst_target_t
#include "xgboost/context.h"     // for Context
#include "xgboost/parameter.h"   // for DMLC_DECLARE_PARAMETER

namespace xgboost {
class Json;

/*! \brief training parameter for regression
 *
 * Should be deprecated, but still used for being compatible with binary IO.
 * Once it's gone, `LearnerModelParam` should handle transforming `base_score`
 * with objective by itself.
 */
struct LearnerModelParamLegacy : public dmlc::Parameter<LearnerModelParamLegacy> {
  /** @brief Global bias/intercept. */
  common::ParamArray<float> base_score{"base_score"};
  /** @brief number of features  */
  bst_feature_t num_feature{0};
  /** @brief number of classes, if it is multi-class classification, 0 otherwise.  */
  std::int32_t num_class{0};
  /**! @brief the version of XGBoost. */
  std::int32_t major_version{std::get<0>(Version::Self())};
  std::int32_t minor_version{std::get<1>(Version::Self())};
  /**
   * @brief Number of target variables.
   */
  bst_target_t num_target{1};
  /**
   * @brief Whether we should calculate the base score from training data.
   *
   *   This is a private parameter as we can't expose it as boolean due to binary model
   *   format. Exposing it as integer creates inconsistency with other parameters.
   *
   *   Automatically disabled when base_score is specifed by user. int32 is used instead
   *   of bool for the ease of serialization.
   */
  std::int32_t boost_from_average{true};

  LearnerModelParamLegacy() = default;

  [[nodiscard]] Json ToJson() const;
  void FromJson(Json const& obj);
  void HandleOldFormat();

  template <typename Container>
  Args UpdateAllowUnknown(Container const& kwargs) {
    // Detect whether user has made their own base score.
    auto has_key = [&kwargs](char const* key) {
      return std::find_if(kwargs.cbegin(), kwargs.cend(),
                          [key](auto const& kv) { return kv.first == key; }) != kwargs.cend();
    };
    if (has_key("base_score")) {
      this->boost_from_average = false;
    }
    return dmlc::Parameter<LearnerModelParamLegacy>::UpdateAllowUnknown(kwargs);
  }

  // The number of outputs of the model.
  [[nodiscard]] bst_target_t OutputLength() const noexcept {
    return std::max({this->num_target, static_cast<bst_target_t>(this->num_class),
                     static_cast<bst_target_t>(1)});
  }

  // Sanity checks
  void Validate(Context const* ctx) const;
  void ValidateLength() const;

  // declare parameters
  DMLC_DECLARE_PARAMETER(LearnerModelParamLegacy) {
    DMLC_DECLARE_FIELD(base_score)
        .describe("Global bias of the model.")
        .set_default(common::ParamArray<float>{"base_score"});
    DMLC_DECLARE_FIELD(num_feature)
        .set_default(0)
        .describe(
            "Number of features in training data, this parameter will be automatically detected by "
            "learner.");
    DMLC_DECLARE_FIELD(num_class).set_default(0).set_lower_bound(0).describe(
        "Number of class option for multi-class classifier. "
        " By default equals 0 and corresponds to binary classifier.");
    DMLC_DECLARE_FIELD(num_target)
        .set_default(1)
        .set_lower_bound(1)
        .describe("Number of output targets. Can be set automatically if not specified.");
    DMLC_DECLARE_FIELD(boost_from_average)
        .set_default(true)
        .describe("Whether we should calculate the base score from training data.");
  }
};
}  // namespace xgboost
