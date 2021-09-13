/*!
 * Copyright 2021 by XGBoost Contributors
 */
#ifndef XGBOOST_DATA_GRADIENT_INDEX_PAGE_SOURCE_H_
#define XGBOOST_DATA_GRADIENT_INDEX_PAGE_SOURCE_H_

#include <memory>
#include <utility>

#include "sparse_page_source.h"
#include "gradient_index.h"

namespace xgboost {
namespace data {
class GradientIndexPageSource : public PageSourceIncMixIn<GHistIndexMatrix> {
  common::HistogramCuts cuts_;
  bool is_dense_;
  int32_t max_bin_per_feat_;

 public:
  GradientIndexPageSource(float missing, int nthreads, bst_feature_t n_features,
                          size_t n_batches, std::shared_ptr<Cache> cache,
                          BatchParam param, common::HistogramCuts cuts,
                          bool is_dense, int32_t max_bin_per_feat,
                          std::shared_ptr<SparsePageSource> source)
      : PageSourceIncMixIn(missing, nthreads, n_features, n_batches, cache),
        cuts_{std::move(cuts)}, is_dense_{is_dense}, max_bin_per_feat_{
                                                         max_bin_per_feat} {
    this->source_ = source;
    this->Fetch();
  }

  void Fetch() final;
};
}      // namespace data
}      // namespace xgboost
#endif  // XGBOOST_DATA_GRADIENT_INDEX_PAGE_SOURCE_H_
