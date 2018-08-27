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
  return source_->info;
}

const MetaInfo& SparsePageDMatrix::Info() const {
  return source_->info;
}

class SparseBatchRowIteratorImpl : public BatchIteratorImpl {
 public:
  explicit SparseBatchRowIteratorImpl(SparsePageSource* source)
      : source_(source) {}
  const SparsePage& operator*() const override { return source_->Value(); }
  void operator++() override { at_end_ = !source_->Next(); }
  bool AtEnd() const override { return at_end_; }
  SparseBatchRowIteratorImpl* Clone() override {
    return new SparseBatchRowIteratorImpl(*this);
  }

 private:
  SparsePageSource* source_{nullptr};
  bool at_end_{ false };
};

class SparseBatchColumnIteratorImpl : public BatchIteratorImpl {
 public:
  explicit SparseBatchColumnIteratorImpl(SparsePageSource* source,
                                         SparsePage* tmp_column_batch,
                                         bool sort)
      : source_(source), tmp_column_batch_(tmp_column_batch), sort_(sort) {}
  const SparsePage& operator*() const override {
    *tmp_column_batch_ = source_->Value().GetTranspose(source_->info.num_col_);
    if (sort_) {
      tmp_column_batch_->SortRows();
    }
    return *tmp_column_batch_;
  }
  void operator++() override { at_end_ = !source_->Next(); }
  bool AtEnd() const override { return at_end_; }
  SparseBatchColumnIteratorImpl* Clone() override {
    return new SparseBatchColumnIteratorImpl(*this);
  }

 private:
  SparsePageSource* source_{nullptr};
  SparsePage*
      tmp_column_batch_;  // Pointer to temporary storage within the dmatrix
  bool sort_;  // If we should return columns in sorted order
  bool at_end_{false};
};

BatchSet SparsePageDMatrix::GetRowBatches() {
  auto cast = dynamic_cast<SparsePageSource*>(source_.get());
  cast->BeforeFirst();
  cast->Next();
  auto begin_iter = BatchIterator(new SparseBatchRowIteratorImpl(cast));
  return BatchSet(begin_iter);
}

BatchSet SparsePageDMatrix::GetSortedColumnBatches() {
  auto cast = dynamic_cast<SparsePageSource*>(source_.get());
  cast->BeforeFirst();
  cast->Next();
  auto begin_iter = BatchIterator(
      new SparseBatchColumnIteratorImpl(cast, &tmp_column_batch_, true));
  return BatchSet(begin_iter);
}

BatchSet SparsePageDMatrix::GetColumnBatches() {
  auto cast = dynamic_cast<SparsePageSource*>(source_.get());
  cast->BeforeFirst();
  cast->Next();
  auto begin_iter = BatchIterator(
      new SparseBatchColumnIteratorImpl(cast, &tmp_column_batch_, false));
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
