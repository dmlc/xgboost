/**
 * Copyright 2021-2023 by XGBoost contributors
 */
#include "../common/hist_util.cuh"
#include "batch_utils.h"  // for CheckEmpty, RegenGHist
#include "ellpack_page.cuh"
#include "sparse_page_dmatrix.h"
#include "sparse_page_source.h"

namespace xgboost::data {
BatchSet<EllpackPage> SparsePageDMatrix::GetEllpackBatches(Context const* ctx,
                                                           const BatchParam& param) {
  CHECK(ctx->IsCUDA());
  CHECK_GE(param.max_bin, 2);
  detail::CheckEmpty(batch_param_, param);
  auto id = MakeCache(this, ".ellpack.page", cache_prefix_, &cache_info_);
  size_t row_stride = 0;
  this->InitializeSparsePage(ctx);
  if (!cache_info_.at(id)->written || detail::RegenGHist(batch_param_, param)) {
    // reinitialize the cache
    cache_info_.erase(id);
    MakeCache(this, ".ellpack.page", cache_prefix_, &cache_info_);
    std::unique_ptr<common::HistogramCuts> cuts;
    cuts.reset(
        new common::HistogramCuts{common::DeviceSketch(ctx->gpu_id, this, param.max_bin, 0)});
    this->InitializeSparsePage(ctx);  // reset after use.

    row_stride = GetRowStride(this);
    this->InitializeSparsePage(ctx);  // reset after use.
    CHECK_NE(row_stride, 0);
    batch_param_ = param;

    auto ft = this->info_.feature_types.ConstDeviceSpan();
    ellpack_page_source_.reset();  // release resources.
    ellpack_page_source_.reset(new EllpackPageSource(
        this->missing_, ctx->Threads(), this->Info().num_col_, this->n_batches_, cache_info_.at(id),
        param, std::move(cuts), this->IsDense(), row_stride, ft, sparse_page_source_, ctx->gpu_id));
  } else {
    CHECK(sparse_page_source_);
    ellpack_page_source_->Reset();
  }

  auto begin_iter = BatchIterator<EllpackPage>(ellpack_page_source_);
  return BatchSet<EllpackPage>(BatchIterator<EllpackPage>(begin_iter));
}
}  // namespace xgboost::data
