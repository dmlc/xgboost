/*!
 * Copyright 2021-2022 by XGBoost Contributors
 */
#ifndef XGBOOST_DATA_GRADIENT_INDEX_PAGE_SOURCE_H_
#define XGBOOST_DATA_GRADIENT_INDEX_PAGE_SOURCE_H_

#include <memory>
#include <utility>

#include "gradient_index.h"
#include "sparse_page_source.h"

namespace xgboost {
namespace data {
class GradientIndexPageSource : public PageSourceIncMixIn<GHistIndexMatrix> {
  common::HistogramCuts cuts_;
  bool is_dense_;
  int32_t max_bin_per_feat_;
  common::Span<FeatureType const> feature_types_;
  double sparse_thresh_;

 public:
  GradientIndexPageSource(float missing, int nthreads, bst_feature_t n_features, size_t n_batches,
                          std::shared_ptr<Cache> cache, BatchParam param,
                          common::HistogramCuts cuts, bool is_dense,
                          common::Span<FeatureType const> feature_types,
                          std::shared_ptr<SparsePageSource> source)
      : PageSourceIncMixIn(missing, nthreads, n_features, n_batches, cache,
                           std::isnan(param.sparse_thresh)),
        cuts_{std::move(cuts)},
        is_dense_{is_dense},
        max_bin_per_feat_{param.max_bin},
        feature_types_{feature_types},
        sparse_thresh_{param.sparse_thresh} {
    this->source_ = source;
    this->Fetch();
  }

  void Fetch() final;
};
}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_GRADIENT_INDEX_PAGE_SOURCE_H_
