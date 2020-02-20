/*!
 * Copyright 2014 by Contributors
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
  if (!ellpack_source_ || batch_param_ != param) {
    ellpack_source_.reset(new EllpackPageSource(this, cache_info_, param));
    batch_param_ = param;
  }
  ellpack_source_->BeforeFirst();
  ellpack_source_->Next();
  auto begin_iter = BatchIterator<EllpackPage>(
      new SparseBatchIteratorImpl<EllpackPageSource, EllpackPage>(ellpack_source_.get()));
  return BatchSet<EllpackPage>(begin_iter);
}

float SparsePageDMatrix::GetColDensity(size_t cidx) {
  // Finds densities if we don't already have them
  if (col_density_.empty()) {
    std::vector<size_t> column_size(this->Info().num_col_);
    for (const auto &batch : this->GetBatches<CSCPage>()) {
      for (auto i = 0u; i < batch.Size(); i++) {
        column_size[i] += batch[i].size();
      }
    }
    col_density_.resize(column_size.size());
    for (auto i = 0u; i < col_density_.size(); i++) {
      size_t nmiss = this->Info().num_row_ - column_size[i];
      col_density_[i] =
          1.0f - (static_cast<float>(nmiss)) / this->Info().num_row_;
    }
  }
  return col_density_.at(cidx);
}

bool SparsePageDMatrix::SingleColBlock() const {
  return false;
}
}  // namespace data
}  // namespace xgboost
#endif  // DMLC_ENABLE_STD_THREAD
