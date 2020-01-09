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
    auto const& page = dynamic_cast<SimpleCSRSource*>(source_.get())->page_;
    column_page_.reset(new CSCPage(page.GetTranspose(source_->info.num_col_)));
  }
  auto begin_iter =
      BatchIterator<CSCPage>(new SimpleBatchIteratorImpl<CSCPage>(column_page_.get()));
  return BatchSet<CSCPage>(begin_iter);
}

BatchSet<SortedCSCPage> SimpleDMatrix::GetSortedColumnBatches() {
  // Sorted column page doesn't exist, generate it
  if (!sorted_column_page_) {
    auto const& page = dynamic_cast<SimpleCSRSource*>(source_.get())->page_;
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

template <typename AdapterT>
SimpleDMatrix::SimpleDMatrix(AdapterT* adapter, float missing, int nthread) {
  // Set number of threads but keep old value so we can reset it after
  const int nthreadmax = omp_get_max_threads();
  if (nthread <= 0) nthread = nthreadmax;
  int nthread_original = omp_get_max_threads();
  omp_set_num_threads(nthread);

  source_.reset(new SimpleCSRSource());
  SimpleCSRSource& mat = *reinterpret_cast<SimpleCSRSource*>(source_.get());
  std::vector<uint64_t> qids;
  uint64_t default_max = std::numeric_limits<uint64_t>::max();
  uint64_t last_group_id = default_max;
  bst_uint group_size = 0;
  auto& offset_vec = mat.page_.offset.HostVector();
  auto& data_vec = mat.page_.data.HostVector();
  uint64_t inferred_num_columns = 0;

  adapter->BeforeFirst();
  // Iterate over batches of input data
  while (adapter->Next()) {
    auto& batch = adapter->Value();
    auto batch_max_columns = mat.page_.Push(batch, missing, nthread);
    inferred_num_columns = std::max(batch_max_columns, inferred_num_columns);
    // Append meta information if available
    if (batch.Labels() != nullptr) {
      auto& labels = mat.info.labels_.HostVector();
      labels.insert(labels.end(), batch.Labels(),
                    batch.Labels() + batch.Size());
    }
    if (batch.Weights() != nullptr) {
      auto& weights = mat.info.weights_.HostVector();
      weights.insert(weights.end(), batch.Weights(),
                     batch.Weights() + batch.Size());
    }
    if (batch.Qid() != nullptr) {
      qids.insert(qids.end(), batch.Qid(), batch.Qid() + batch.Size());
      // get group
      for (size_t i = 0; i < batch.Size(); ++i) {
        const uint64_t cur_group_id = batch.Qid()[i];
        if (last_group_id == default_max || last_group_id != cur_group_id) {
          mat.info.group_ptr_.push_back(group_size);
        }
        last_group_id = cur_group_id;
        ++group_size;
      }
    }
  }

  if (last_group_id != default_max) {
    if (group_size > mat.info.group_ptr_.back()) {
      mat.info.group_ptr_.push_back(group_size);
    }
  }

  // Deal with empty rows/columns if necessary
  if (adapter->NumColumns() == kAdapterUnknownSize) {
    mat.info.num_col_ = inferred_num_columns;
  } else {
    mat.info.num_col_ = adapter->NumColumns();
  }
  // Synchronise worker columns
  rabit::Allreduce<rabit::op::Max>(&mat.info.num_col_, 1);

  if (adapter->NumRows() == kAdapterUnknownSize) {
    mat.info.num_row_ = offset_vec.size() - 1;
  } else {
    if (offset_vec.empty()) {
      offset_vec.emplace_back(0);
    }

    while (offset_vec.size() - 1 < adapter->NumRows()) {
      offset_vec.emplace_back(offset_vec.back());
    }
    mat.info.num_row_ = adapter->NumRows();
  }
  mat.info.num_nonzero_ = data_vec.size();
  omp_set_num_threads(nthread_original);
}

template SimpleDMatrix::SimpleDMatrix(DenseAdapter* adapter, float missing,
                                     int nthread);
template SimpleDMatrix::SimpleDMatrix(CSRAdapter* adapter, float missing,
                                     int nthread);
template SimpleDMatrix::SimpleDMatrix(CSCAdapter* adapter, float missing,
                                     int nthread);
template SimpleDMatrix::SimpleDMatrix(DataTableAdapter* adapter, float missing,
                                     int nthread);
template SimpleDMatrix::SimpleDMatrix(FileAdapter* adapter, float missing,
                                     int nthread);
}  // namespace data
}  // namespace xgboost
