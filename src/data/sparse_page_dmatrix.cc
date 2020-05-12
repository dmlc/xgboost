/*!
 * Copyright 2014-2020 by Contributors
 * \file sparse_page_dmatrix.cc
 * \brief The external memory version of Page Iterator.
 * \author Tianqi Chen
 */
#include <dmlc/base.h>
#include <dmlc/timer.h>

#if DMLC_ENABLE_STD_THREAD
#include "./sparse_page_dmatrix.h"

#include "./simple_batch_iterator.h"

namespace xgboost {
namespace data {

MetaInfo& SparsePageDMatrix::Info() {
  return row_source_->info;
}

const MetaInfo& SparsePageDMatrix::Info() const {
  return row_source_->info;
}

BatchSet<SparsePage> SparsePageDMatrix::GetRowBatches() {
  return row_source_->GetBatchSet();
}

BatchSet<CSCPage> SparsePageDMatrix::GetColumnBatches() {
  // Lazily instantiate
  if (!column_source_) {
    column_source_.reset(new CSCPageSource(this, cache_info_));
  }
  return column_source_->GetBatchSet();
}

BatchSet<SortedCSCPage> SparsePageDMatrix::GetSortedColumnBatches() {
  // Lazily instantiate
  if (!sorted_column_source_) {
    sorted_column_source_.reset(new SortedCSCPageSource(this, cache_info_));
  }
  return sorted_column_source_->GetBatchSet();
}

BatchSet<EllpackPage> SparsePageDMatrix::GetEllpackBatches(const BatchParam& param) {
  CHECK_GE(param.gpu_id, 0);
  CHECK_GE(param.max_bin, 2);
  // Lazily instantiate
  if (!ellpack_source_ || (batch_param_ != param && param != BatchParam{})) {
    ellpack_source_.reset(new EllpackPageSource(this, cache_info_, param));
    batch_param_ = param;
  }
  return ellpack_source_->GetBatchSet();
}

}  // namespace data
}  // namespace xgboost
#endif  // DMLC_ENABLE_STD_THREAD
