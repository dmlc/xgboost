/*!
 * Copyright 2014 by Contributors
 * \file sparse_page_dmatrix.cc
 * \brief The external memory version of Page Iterator.
 * \author Tianqi Chen
 */
#include <dmlc/base.h>
#include <dmlc/timer.h>
#include <xgboost/logging.h>
#include <memory>

#if DMLC_ENABLE_STD_THREAD
#include "./sparse_page_dmatrix.h"
#include "../common/random.h"

namespace xgboost {
namespace data {

MetaInfo& SparsePageDMatrix::Info() {
  return row_source_->info;
}

const MetaInfo& SparsePageDMatrix::Info() const {
  return row_source_->info;
}

class SparseBatchIteratorImpl : public BatchIteratorImpl {
 public:
  explicit SparseBatchIteratorImpl(SparsePageSource* source) : source_(source) {
    CHECK(source_ != nullptr);
  }
  const SparsePage& operator*() const override { return source_->Value(); }
  void operator++() override { at_end_ = !source_->Next(); }
  bool AtEnd() const override { return at_end_; }
  SparseBatchIteratorImpl* Clone() override {
    return new SparseBatchIteratorImpl(*this);
  }

 private:
  SparsePageSource* source_{nullptr};
  bool at_end_{ false };
};

BatchSet SparsePageDMatrix::GetRowBatches() {
  auto cast = dynamic_cast<SparsePageSource*>(row_source_.get());
  cast->BeforeFirst();
  cast->Next();
  auto begin_iter = BatchIterator(new SparseBatchIteratorImpl(cast));
  return BatchSet(begin_iter);
}

BatchSet SparsePageDMatrix::GetSortedColumnBatches() {
  // Lazily instantiate
  if (!sorted_column_source_) {
    SparsePageSource::CreateColumnPage(this, cache_info_, true);
    sorted_column_source_.reset(
        new SparsePageSource(cache_info_, ".sorted.col.page"));
  }
  sorted_column_source_->BeforeFirst();
  sorted_column_source_->Next();
  auto begin_iter =
      BatchIterator(new SparseBatchIteratorImpl(sorted_column_source_.get()));
  return BatchSet(begin_iter);
}

BatchSet SparsePageDMatrix::GetColumnBatches() {
  // Lazily instantiate
  if (!column_source_) {
    SparsePageSource::CreateColumnPage(this, cache_info_, false);
    column_source_.reset(new SparsePageSource(cache_info_, ".col.page"));
  }
  column_source_->BeforeFirst();
  column_source_->Next();
  auto begin_iter =
      BatchIterator(new SparseBatchIteratorImpl(column_source_.get()));
  return BatchSet(begin_iter);
}

float SparsePageDMatrix::GetColDensity(size_t cidx) {
  // Finds densities if we don't already have them
  if (col_density_.empty()) {
    std::vector<size_t> column_size(this->Info().num_col_);
    for (const auto &batch : this->GetColumnBatches()) {
      for (int i = 0; i < batch.Size(); i++) {
        column_size[i] += batch[i].size();
      }
    }
    col_density_.resize(column_size.size());
    for (int i = 0; i < col_density_.size(); i++) {
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
#endif
