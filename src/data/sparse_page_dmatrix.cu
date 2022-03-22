/*!
 * Copyright 2021 XGBoost contributors
 */
#include "sparse_page_source.h"
#include "../common/hist_util.cuh"
#include "ellpack_page.cuh"
#include "sparse_page_dmatrix.h"

namespace xgboost {
namespace data {
BatchSet<EllpackPage> SparsePageDMatrix::GetEllpackBatches(const BatchParam& param) {
  CHECK_GE(param.gpu_id, 0);
  CHECK_GE(param.max_bin, 2);
  if (!(batch_param_ != BatchParam{})) {
    CHECK(param != BatchParam{}) << "Batch parameter is not initialized.";
  }
  auto id = MakeCache(this, ".ellpack.page", cache_prefix_, &cache_info_);
  size_t row_stride = 0;
  this->InitializeSparsePage();
  if (!cache_info_.at(id)->written || RegenGHist(batch_param_, param)) {
    // reinitialize the cache
    cache_info_.erase(id);
    MakeCache(this, ".ellpack.page", cache_prefix_, &cache_info_);
    std::unique_ptr<common::HistogramCuts> cuts;
    cuts.reset(new common::HistogramCuts{
        common::DeviceSketch(param.gpu_id, this, param.max_bin, 0)});
    this->InitializeSparsePage();  // reset after use.

    row_stride = GetRowStride(this);
    this->InitializeSparsePage();  // reset after use.
    CHECK_NE(row_stride, 0);
    batch_param_ = param;

    auto ft = this->info_.feature_types.ConstDeviceSpan();
    ellpack_page_source_.reset();  // release resources.
    ellpack_page_source_.reset(new EllpackPageSource(
        this->missing_, this->ctx_.Threads(), this->Info().num_col_,
        this->n_batches_, cache_info_.at(id), param, std::move(cuts),
        this->IsDense(), row_stride, ft, sparse_page_source_));
  } else {
    CHECK(sparse_page_source_);
    ellpack_page_source_->Reset();
  }

  auto begin_iter = BatchIterator<EllpackPage>(ellpack_page_source_);
  return BatchSet<EllpackPage>(BatchIterator<EllpackPage>(begin_iter));
}
}  // namespace data
}  // namespace xgboost
