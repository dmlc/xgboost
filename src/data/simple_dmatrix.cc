/*!
 * Copyright 2014 by Contributors
 * \file simple_dmatrix.cc
 * \brief the input data structure for gradient boosting
 * \author Tianqi Chen
 */
#include "./simple_dmatrix.h"
#include <xgboost/data.h>
#include "../common/random.h"

namespace xgboost {
namespace data {
MetaInfo& SimpleDMatrix::Info() { return source_->info; }

const MetaInfo& SimpleDMatrix::Info() const { return source_->info; }

float SimpleDMatrix::GetColDensity(size_t cidx) {
  size_t column_size = 0;
  // Use whatever version of column batches already exists
  if (sorted_column_page_) {
    auto batch = this->GetBatches<SortedCSCPage>();
    column_size = (*batch.begin())[cidx].size();
  } else {
    auto batch = this->GetBatches<CSCPage>();
    column_size = (*batch.begin())[cidx].size();
  }

  size_t nmiss = this->Info().num_row_ - column_size;
  return 1.0f - (static_cast<float>(nmiss)) / this->Info().num_row_;
}

template<typename T>
class SimpleBatchIteratorImpl : public BatchIteratorImpl<T> {
 public:
  explicit SimpleBatchIteratorImpl(T* page) : page_(page) {}
  T& operator*() override {
    CHECK(page_ != nullptr);
    return *page_;
  }
  const T& operator*() const override {
    CHECK(page_ != nullptr);
    return *page_;
  }
  void operator++() override { page_ = nullptr; }
  bool AtEnd() const override { return page_ == nullptr; }

 private:
  T* page_{nullptr};
};

BatchSet<SparsePage> SimpleDMatrix::GetRowBatches() {
  // since csr is the default data structure so `source_` is always available.
  auto cast = dynamic_cast<SimpleCSRSource*>(source_.get());
  auto begin_iter = BatchIterator<SparsePage>(
      new SimpleBatchIteratorImpl<SparsePage>(&(cast->page_)));
  return BatchSet<SparsePage>(begin_iter);
}

BatchSet<CSCPage> SimpleDMatrix::GetColumnBatches() {
  // column page doesn't exist, generate it
  if (!column_page_) {
    auto page = dynamic_cast<SimpleCSRSource*>(source_.get())->page_;
    column_page_.reset(new CSCPage(page.GetTranspose(source_->info.num_col_)));
  }
  auto begin_iter =
      BatchIterator<CSCPage>(new SimpleBatchIteratorImpl<CSCPage>(column_page_.get()));
  return BatchSet<CSCPage>(begin_iter);
}

BatchSet<SortedCSCPage> SimpleDMatrix::GetSortedColumnBatches() {
  // Sorted column page doesn't exist, generate it
  if (!sorted_column_page_) {
    auto page = dynamic_cast<SimpleCSRSource*>(source_.get())->page_;
    sorted_column_page_.reset(
        new SortedCSCPage(page.GetTranspose(source_->info.num_col_)));
    sorted_column_page_->SortRows();
  }
  auto begin_iter = BatchIterator<SortedCSCPage>(
      new SimpleBatchIteratorImpl<SortedCSCPage>(sorted_column_page_.get()));
  return BatchSet<SortedCSCPage>(begin_iter);
}

bool SimpleDMatrix::SingleColBlock() const { return true; }
}  // namespace data
}  // namespace xgboost
