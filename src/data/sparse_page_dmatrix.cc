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

template<typename S, typename T>
class SparseBatchIteratorImpl : public BatchIteratorImpl<T> {
 public:
  explicit SparseBatchIteratorImpl(S* source) : source_(source) {
    CHECK(source_ != nullptr);
  }
  T& operator*() override { return source_->Value(); }
  const T& operator*() const override { return source_->Value(); }
  void operator++() override { at_end_ = !source_->Next(); }
  bool AtEnd() const override { return at_end_; }

 private:
  S* source_{nullptr};
  bool at_end_{ false };
};

BatchSet<SparsePage> SparsePageDMatrix::GetRowBatches() {
  auto cast = dynamic_cast<SparsePageSource<SparsePage>*>(row_source_.get());
  CHECK(cast);
  cast->BeforeFirst();
  cast->Next();
  auto begin_iter = BatchIterator<SparsePage>(
      new SparseBatchIteratorImpl<SparsePageSource<SparsePage>, SparsePage>(cast));
  return BatchSet<SparsePage>(begin_iter);
}

BatchSet<CSCPage> SparsePageDMatrix::GetColumnBatches() {
  // Lazily instantiate
  if (!column_source_) {
    SparsePageSource<SparsePage>::CreateColumnPage(this, cache_info_, false);
    column_source_.reset(new SparsePageSource<CSCPage>(cache_info_, ".col.page"));
  }
  column_source_->BeforeFirst();
  column_source_->Next();
  auto begin_iter = BatchIterator<CSCPage>(
      new SparseBatchIteratorImpl<SparsePageSource<CSCPage>, CSCPage>(column_source_.get()));
  return BatchSet<CSCPage>(begin_iter);
}

BatchSet<SortedCSCPage> SparsePageDMatrix::GetSortedColumnBatches() {
  // Lazily instantiate
  if (!sorted_column_source_) {
    SparsePageSource<SparsePage>::CreateColumnPage(this, cache_info_, true);
    sorted_column_source_.reset(
        new SparsePageSource<SortedCSCPage>(cache_info_, ".sorted.col.page"));
  }
  sorted_column_source_->BeforeFirst();
  sorted_column_source_->Next();
  auto begin_iter = BatchIterator<SortedCSCPage>(
      new SparseBatchIteratorImpl<SparsePageSource<SortedCSCPage>, SortedCSCPage>(
          sorted_column_source_.get()));
  return BatchSet<SortedCSCPage>(begin_iter);
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
