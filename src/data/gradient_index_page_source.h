/**
 * Copyright 2021-2024, XGBoost Contributors
 */
#ifndef XGBOOST_DATA_GRADIENT_INDEX_PAGE_SOURCE_H_
#define XGBOOST_DATA_GRADIENT_INDEX_PAGE_SOURCE_H_

#include <cmath>    // for isnan
#include <cstdint>  // for int32_t
#include <memory>   // for shared_ptr
#include <utility>  // for move
#include <vector>   // for vector

#include "../common/hist_util.h"    // for HistogramCuts
#include "gradient_index.h"         // for GHistIndexMatrix
#include "gradient_index_format.h"  // for GHistIndexRawFormat
#include "sparse_page_source.h"     // for PageSourceIncMixIn
#include "xgboost/base.h"           // for bst_feature_t
#include "xgboost/data.h"           // for BatchParam, FeatureType
#include "xgboost/span.h"           // for Span

namespace xgboost::data {
/**
 * @brief Policy for creating ghist index format. The storage is default (disk).
 */
template <typename S>
class GHistIndexFormatPolicy {
 protected:
  common::HistogramCuts cuts_;

 public:
  using FormatT = SparsePageFormat<GHistIndexMatrix>;

 public:
  [[nodiscard]] auto CreatePageFormat(BatchParam const&) const {
    std::unique_ptr<FormatT> fmt{new GHistIndexRawFormat{cuts_}};
    return fmt;
  }

  void SetCuts(common::HistogramCuts cuts) { std::swap(cuts_, cuts); }
};

class GradientIndexPageSource
    : public PageSourceIncMixIn<
          GHistIndexMatrix, DefaultFormatStreamPolicy<GHistIndexMatrix, GHistIndexFormatPolicy>> {
  bool is_dense_;
  std::int32_t max_bin_per_feat_;
  common::Span<FeatureType const> feature_types_;
  double sparse_thresh_;

 public:
  GradientIndexPageSource(float missing, std::int32_t nthreads, bst_feature_t n_features,
                          size_t n_batches, std::shared_ptr<Cache> cache, BatchParam param,
                          common::HistogramCuts cuts, bool is_dense,
                          common::Span<FeatureType const> feature_types,
                          std::shared_ptr<SparsePageSource> source)
      : PageSourceIncMixIn(missing, nthreads, n_features, n_batches, cache,
                           std::isnan(param.sparse_thresh)),
        is_dense_{is_dense},
        max_bin_per_feat_{param.max_bin},
        feature_types_{feature_types},
        sparse_thresh_{param.sparse_thresh} {
    this->source_ = source;
    this->SetCuts(std::move(cuts));
    this->Fetch();
  }

  void Fetch() final;
};

class ExtGradientIndexPageSource
    : public ExtQantileSourceMixin<
          GHistIndexMatrix, DefaultFormatStreamPolicy<GHistIndexMatrix, GHistIndexFormatPolicy>> {
  BatchParam p_;

  Context const* ctx_;
  DMatrixProxy* proxy_;
  MetaInfo* info_;

  common::Span<FeatureType const> feature_types_;
  std::vector<bst_idx_t> base_rows_;

 public:
  ExtGradientIndexPageSource(
      Context const* ctx, float missing, MetaInfo* info, bst_idx_t n_batches,
      std::shared_ptr<Cache> cache, BatchParam param, common::HistogramCuts cuts,
      std::shared_ptr<DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>> source,
      DMatrixProxy* proxy, std::vector<bst_idx_t> base_rows)
      : ExtQantileSourceMixin{missing,   ctx->Threads(), static_cast<bst_feature_t>(info->num_col_),
                              n_batches, source,         cache},
        p_{std::move(param)},
        ctx_{ctx},
        proxy_{proxy},
        info_{info},
        feature_types_{info_->feature_types.ConstHostSpan()},
        base_rows_{std::move(base_rows)} {
    this->SetCuts(std::move(cuts));
    this->Fetch();
  }

  void Fetch() final;
};
}  // namespace xgboost::data
#endif  // XGBOOST_DATA_GRADIENT_INDEX_PAGE_SOURCE_H_
