/**
 * Copyright 2023 by XGBoost contributors
 */
#ifndef XGBOOST_COMMON_RANKING_UTILS_H_
#define XGBOOST_COMMON_RANKING_UTILS_H_

#include <cstddef>                // std::size_t
#include <cstdint>                // std::uint32_t
#include <string>                 // std::string

#include "xgboost/string_view.h"  // StringView

namespace xgboost {
namespace ltr {
/**
 * \brief Construct name for ranking metric given parameters.
 *
 * \param [in] name   Null terminated string for metric name
 * \param [in] param  Null terminated string for parameter like the `3-` in `ndcg@3-`.
 * \param [out] topn  Top n documents parsed from param. Unchanged if it's not specified.
 * \param [out] minus Whether we should turn the score into loss. Unchanged if it's not
 *                    specified.
 *
 * \return The name of the metric.
 */
std::string MakeMetricName(StringView name, StringView param, std::uint32_t* topn, bool* minus);
}  // namespace ltr
}  // namespace xgboost
#endif  // XGBOOST_COMMON_RANKING_UTILS_H_
