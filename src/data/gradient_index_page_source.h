/*!
 * Copyright 2021 by XGBoost Contributors
 */

#ifndef XGBOOST_DATA_GRADIENT_INDEX_PAGE_SOURCE_H_
#define XGBOOST_DATA_GRADIENT_INDEX_PAGE_SOURCE_H_

#include "sparse_page_source.h"
#include "gradient_index.h"

namespace xgboost {
namespace data {
class GradientIndexPageSource : public PageSourceIncMixIn<GHistIndexMatrix> {
  common::HistogramCuts cuts_;

 public:
  GradientIndexPageSource(
      float missing, int nthreads, bst_feature_t n_features, size_t n_batches,
      std::shared_ptr<Cache> cache, BatchParam param,
      common::HistogramCuts cuts, bool is_dense,
      std::shared_ptr<SparsePageSource> source)
      : PageSourceIncMixIn(missing, nthreads, n_features, n_batches, cache),
        cuts_{std::move(cuts)} {
    this->source_ = source;
    this->Fetch();
  }

  void Fetch() final;
};
}      // namespace data
}      // namespace xgboost
#endif  // XGBOOST_DATA_GRADIENT_INDEX_PAGE_SOURCE_H_
