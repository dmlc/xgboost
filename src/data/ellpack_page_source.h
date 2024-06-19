/**
 * Copyright 2019-2024, XGBoost Contributors
 */

#ifndef XGBOOST_DATA_ELLPACK_PAGE_SOURCE_H_
#define XGBOOST_DATA_ELLPACK_PAGE_SOURCE_H_

#include <cstdint>  // for int32_t
#include <memory>   // for shared_ptr
#include <utility>  // for move

#include "../common/hist_util.h"      // for HistogramCuts
#include "ellpack_page.h"             // for EllpackPage
#include "ellpack_page_raw_format.h"  // for EllpackPageRawFormat
#include "sparse_page_source.h"       // for PageSourceIncMixIn
#include "xgboost/base.h"             // for bst_idx_t
#include "xgboost/context.h"          // for DeviceOrd
#include "xgboost/data.h"             // for BatchParam
#include "xgboost/span.h"             // for Span

namespace xgboost::data {
class EllpackPageSource : public PageSourceIncMixIn<EllpackPage> {
  bool is_dense_;
  bst_idx_t row_stride_;
  BatchParam param_;
  common::Span<FeatureType const> feature_types_;
  std::shared_ptr<common::HistogramCuts const> cuts_;
  DeviceOrd device_;

 protected:
  [[nodiscard]] SparsePageFormat<EllpackPage>* CreatePageFormat() const override {
    cuts_->SetDevice(this->device_);
    return new EllpackPageRawFormat{cuts_};
  }

 public:
  EllpackPageSource(float missing, std::int32_t nthreads, bst_feature_t n_features,
                    size_t n_batches, std::shared_ptr<Cache> cache, BatchParam param,
                    std::shared_ptr<common::HistogramCuts const> cuts, bool is_dense,
                    bst_idx_t row_stride, common::Span<FeatureType const> feature_types,
                    std::shared_ptr<SparsePageSource> source, DeviceOrd device)
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
  (void)(device_);
  common::AssertGPUSupport();
}
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace xgboost::data

#endif  // XGBOOST_DATA_ELLPACK_PAGE_SOURCE_H_
