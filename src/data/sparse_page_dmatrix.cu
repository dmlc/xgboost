/**
 * Copyright 2021-2024, XGBoost contributors
 */
#include <memory>  // for unique_ptr

#include "../common/hist_util.cuh"
#include "../common/hist_util.h"  // for HistogramCuts
#include "batch_utils.h"          // for CheckEmpty, RegenGHist
#include "ellpack_page.cuh"
#include "sparse_page_dmatrix.h"
#include "xgboost/context.h"  // for Context
#include "xgboost/data.h"     // for BatchParam

namespace xgboost::data {
BatchSet<EllpackPage> SparsePageDMatrix::GetEllpackBatches(Context const* ctx,
                                                           const BatchParam& param) {
  CHECK(ctx->IsCUDA());
  if (param.Initialized()) {
    CHECK_GE(param.max_bin, 2);
  }
  detail::CheckEmpty(batch_param_, param);
  auto id = MakeCache(this, ".ellpack.page", cache_prefix_, &cache_info_);
  size_t row_stride = 0;
  if (!cache_info_.at(id)->written || detail::RegenGHist(batch_param_, param)) {
    this->InitializeSparsePage(ctx);
    // reinitialize the cache
    cache_info_.erase(id);
    MakeCache(this, ".ellpack.page", cache_prefix_, &cache_info_);
    std::unique_ptr<common::HistogramCuts> cuts;
    if (!param.hess.empty()) {
      cuts = std::make_unique<common::HistogramCuts>(
          common::DeviceSketchWithHessian(ctx, this, param.max_bin, param.hess));
    } else {
      cuts =
          std::make_unique<common::HistogramCuts>(common::DeviceSketch(ctx, this, param.max_bin));
    }
    this->InitializeSparsePage(ctx);  // reset after use.

    row_stride = GetRowStride(this);
    this->InitializeSparsePage(ctx);  // reset after use.
    CHECK_NE(row_stride, 0);
    batch_param_ = param;

    auto ft = this->info_.feature_types.ConstDeviceSpan();
    ellpack_page_source_.reset();  // make sure resource is released before making new ones.
    ellpack_page_source_ = std::make_shared<EllpackPageSource>(
        this->missing_, ctx->Threads(), this->Info().num_col_, this->n_batches_, cache_info_.at(id),
        param, std::move(cuts), this->IsDense(), row_stride, ft, sparse_page_source_,
        ctx->Device());
  } else {
    CHECK(sparse_page_source_);
    ellpack_page_source_->Reset();
  }

  return BatchSet{BatchIterator<EllpackPage>{this->ellpack_page_source_}};
}
}  // namespace xgboost::data
