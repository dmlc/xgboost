/*!
 * Copyright 2014 by Contributors
 * \file simple_dmatrix.cc
 * \brief the input data structure for gradient boosting
 * \author Tianqi Chen
 */
#include "./simple_dmatrix.h"
#include <xgboost/data.h>
#include "./simple_batch_iterator.h"
#include "../common/random.h"
#include "../common/math.h"
#include "../common/group_data.h"

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

BatchSet<EllpackPage> SimpleDMatrix::GetEllpackBatches(const BatchParam& param) {
  CHECK_GE(param.gpu_id, 0);
  CHECK_GE(param.max_bin, 2);
  // ELLPACK page doesn't exist, generate it
  if (!ellpack_page_) {
    ellpack_page_.reset(new EllpackPage(this, param));
  }
  auto begin_iter =
      BatchIterator<EllpackPage>(new SimpleBatchIteratorImpl<EllpackPage>(ellpack_page_.get()));
  return BatchSet<EllpackPage>(begin_iter);
}

bool SimpleDMatrix::SingleColBlock() const { return true; }

  SimpleDMatrix::SimpleDMatrix(const ExternalDataAdapter* adapter, int nthread) {
  const int nthreadmax = omp_get_max_threads();
    if (nthread <= 0) nthread = nthreadmax;
  int nthread_orig = omp_get_max_threads();
  omp_set_num_threads(nthread);
  source_.reset(new SimpleCSRSource());
  SimpleCSRSource& mat = *reinterpret_cast<SimpleCSRSource*>(source_.get());
  auto& offset_vec = mat.page_.offset.HostVector();
  auto& data_vec = mat.page_.data.HostVector();
  common::ParallelGroupBuilder<
      Entry, std::remove_reference<decltype(offset_vec)>::type::value_type>
      builder(&offset_vec, &data_vec);
  builder.InitBudget(adapter->GetNumRows(), nthread);
  size_t num_batches = adapter->Size();
#pragma omp parallel for schedule(static)
  for (omp_ulong i = 0; i < static_cast<omp_ulong>(num_batches);
       ++i) {  // NOLINT(*)
    int tid = omp_get_thread_num();
    auto batch = (*adapter)[i];
    for (auto j = 0ull; j < batch->Size(); j++) {
      auto element = batch->GetElement(j);
      if (!common::CheckNAN(element.value)) {
        builder.AddBudget(element.row_idx, tid);
      }
    }
  }
  builder.InitStorage();
#pragma omp parallel for schedule(static)
  for (omp_ulong i = 0; i < static_cast<omp_ulong>(num_batches);
       ++i) {  // NOLINT(*)
    int tid = omp_get_thread_num();
    auto batch = (*adapter)[i];
    for (auto j = 0ull; j < batch->Size(); j++) {
      auto element = batch->GetElement(j);
      builder.Push(element.row_idx, Entry(element.column_idx, element.value),
                   tid);
    }
  }
  mat.info.num_row_ = mat.page_.offset.Size() - 1;
  if (adapter->GetNumRows() > 0) {
    CHECK_LE(mat.info.num_row_, adapter->GetNumRows());
    // provision for empty rows at the bottom of matrix
    auto& offset_vec = mat.page_.offset.HostVector();
    for (uint64_t i = mat.info.num_row_;
         i < static_cast<uint64_t>(adapter->GetNumRows()); ++i) {
      offset_vec.push_back(offset_vec.back());
    }
    mat.info.num_row_ = adapter->GetNumRows();
    CHECK_EQ(mat.info.num_row_, offset_vec.size() - 1);  // sanity check
  }
  mat.info.num_col_ = adapter->GetNumFeatures();
  mat.info.num_nonzero_ = adapter->GetNumElements();
  omp_set_num_threads(nthread_orig);
}

}  // namespace data
}  // namespace xgboost
