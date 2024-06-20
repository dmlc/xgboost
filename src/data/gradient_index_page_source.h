/**
 * Copyright 2021-2024, XGBoost Contributors
 */
#ifndef XGBOOST_DATA_GRADIENT_INDEX_PAGE_SOURCE_H_
#define XGBOOST_DATA_GRADIENT_INDEX_PAGE_SOURCE_H_

#include <cmath>    // for isnan
#include <cstdint>  // for int32_t
#include <memory>   // for shared_ptr
#include <utility>  // for move

#include "../common/hist_util.h"    // for HistogramCuts
#include "gradient_index.h"         // for GHistIndexMatrix
#include "gradient_index_format.h"  // for GHistIndexRawFormat
#include "sparse_page_source.h"     // for PageSourceIncMixIn
#include "xgboost/base.h"           // for bst_feature_t
#include "xgboost/data.h"           // for BatchParam, FeatureType
#include "xgboost/span.h"           // for Span

namespace xgboost {
namespace data {
class GradientIndexPageSource : public PageSourceIncMixIn<GHistIndexMatrix> {
  common::HistogramCuts cuts_;
  bool is_dense_;
  std::int32_t max_bin_per_feat_;
  common::Span<FeatureType const> feature_types_;
  double sparse_thresh_;

 protected:
  [[nodiscard]] SparsePageFormat<GHistIndexMatrix>* CreatePageFormat() const override {
    return new GHistIndexRawFormat{cuts_};
  }

 public:
  GradientIndexPageSource(float missing, std::int32_t nthreads, bst_feature_t n_features,
                          size_t n_batches, std::shared_ptr<Cache> cache, BatchParam param,
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
