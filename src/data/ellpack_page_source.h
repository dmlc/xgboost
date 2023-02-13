/**
 * Copyright 2019-2023, XGBoost Contributors
 */

#ifndef XGBOOST_DATA_ELLPACK_PAGE_SOURCE_H_
#define XGBOOST_DATA_ELLPACK_PAGE_SOURCE_H_

#include <xgboost/data.h>
#include <memory>
#include <string>
#include <utility>

#include "../common/common.h"
#include "../common/hist_util.h"
#include "sparse_page_source.h"

namespace xgboost {
namespace data {

class EllpackPageSource : public PageSourceIncMixIn<EllpackPage> {
  bool is_dense_;
  size_t row_stride_;
  BatchParam param_;
  common::Span<FeatureType const> feature_types_;
  std::unique_ptr<common::HistogramCuts> cuts_;
  std::int32_t device_;

 public:
  EllpackPageSource(float missing, int nthreads, bst_feature_t n_features, size_t n_batches,
                    std::shared_ptr<Cache> cache, BatchParam param,
                    std::unique_ptr<common::HistogramCuts> cuts, bool is_dense, size_t row_stride,
                    common::Span<FeatureType const> feature_types,
                    std::shared_ptr<SparsePageSource> source, std::int32_t device)
      : PageSourceIncMixIn(missing, nthreads, n_features, n_batches, cache, false),
        is_dense_{is_dense},
        row_stride_{row_stride},
        param_{std::move(param)},
        feature_types_{feature_types},
        cuts_{std::move(cuts)},
        device_{device} {
    this->source_ = source;
    this->Fetch();
  }

  void Fetch() final;
};

#if !defined(XGBOOST_USE_CUDA)
inline void EllpackPageSource::Fetch() {
  // silent the warning about unused variables.
  (void)(row_stride_);
  (void)(is_dense_);
  common::AssertGPUSupport();
}
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace data
}  // namespace xgboost

#endif  // XGBOOST_DATA_ELLPACK_PAGE_SOURCE_H_
