/**
 * Copyright 2023 by XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_RANKING_UTILS_CUH_
#define XGBOOST_COMMON_RANKING_UTILS_CUH_

#include <cstddef>            // for size_t

#include "ranking_utils.h"    // for LambdaRankParam
#include "xgboost/base.h"     // for bst_group_t, XGBOOST_DEVICE
#include "xgboost/context.h"  // for Context
#include "xgboost/linalg.h"   // for VectorView
#include "xgboost/span.h"     // for Span

namespace xgboost {
namespace ltr {
namespace cuda_impl {
void CalcQueriesDCG(Context const *ctx, linalg::VectorView<float const> d_labels,
                    common::Span<std::size_t const> d_sorted_idx, bool exp_gain,
                    common::Span<bst_group_t const> d_group_ptr, std::size_t k,
                    linalg::VectorView<double> out_dcg);

void CalcQueriesInvIDCG(Context const *ctx, linalg::VectorView<float const> d_labels,
                        common::Span<bst_group_t const> d_group_ptr,
                        linalg::VectorView<double> out_inv_IDCG, ltr::LambdaRankParam const &p);

// Functions for creating number of threads for CUDA, and getting back the number of pairs
// from the number of threads.
XGBOOST_DEVICE __forceinline__ std::size_t ThreadsForMean(std::size_t group_size,
                                                          std::size_t n_pairs) {
  return group_size * n_pairs;
}
XGBOOST_DEVICE __forceinline__ std::size_t PairsForGroup(std::size_t n_threads,
                                                         std::size_t group_size) {
  return n_threads / group_size;
}
}  // namespace cuda_impl
}  // namespace ltr
}  // namespace xgboost
#endif  // XGBOOST_COMMON_RANKING_UTILS_CUH_
