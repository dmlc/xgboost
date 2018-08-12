/*!
 * Copyright 2014 by Contributors
 * \file simple_dmatrix.cc
 * \brief the input data structure for gradient boosting
 * \author Tianqi Chen
 */
#include <xgboost/data.h>
#include <limits>
#include <algorithm>
#include <vector>
#include "./simple_dmatrix.h"
#include "../common/random.h"
#include "../common/group_data.h"

namespace xgboost {
namespace data {

bool SimpleDMatrix::ColBatchIter::Next() {
  if (data_ >= 1) return false;
  data_ += 1;
  return true;
}

  dmlc::DataIter<SparsePage>* SimpleDMatrix::ColIterator() {
  col_iter_.BeforeFirst();
  return &col_iter_;
}

void SimpleDMatrix::InitColAccess(
  size_t max_row_perbatch, bool sorted) {
  if (this->HaveColAccess(sorted)) return;
  col_iter_.sorted_ = sorted;
  col_iter_.column_page_.reset(new SparsePage());
  this->MakeOneBatch(col_iter_.column_page_.get(), sorted);
}

// internal function to make one batch from row iter.
void SimpleDMatrix::MakeOneBatch(SparsePage* pcol, bool sorted) {
  // clear rowset
  buffered_rowset_.Clear();
  // bit map
  const int nthread = omp_get_max_threads();
  pcol->Clear();
  common::ParallelGroupBuilder<Entry>
      builder(&pcol->offset, &pcol->data);
  builder.InitBudget(Info().num_col_, nthread);
  // start working
  auto iter = this->RowIterator();
  iter->BeforeFirst();
  while (iter->Next()) {
    const  auto& batch = iter->Value();
    long batch_size = static_cast<long>(batch.Size()); // NOLINT(*)
    for (long i = 0; i < batch_size; ++i) { // NOLINT(*)
      auto ridx = static_cast<bst_uint>(batch.base_rowid + i);
      buffered_rowset_.PushBack(ridx);
    }
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < batch_size; ++i) { // NOLINT(*)
      int tid = omp_get_thread_num();
      auto inst = batch[i];
      for (bst_uint j = 0; j < inst.length; ++j) {
        builder.AddBudget(inst[j].index, tid);
      }
    }
  }
  builder.InitStorage();

  iter->BeforeFirst();
  while (iter->Next()) {
     auto &batch = iter->Value();
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < static_cast<long>(batch.Size()); ++i) { // NOLINT(*)
      int tid = omp_get_thread_num();
      auto inst = batch[i];
      for (bst_uint j = 0; j < inst.length; ++j) {
        builder.Push(
            inst[j].index,
            Entry(static_cast<bst_uint>(batch.base_rowid + i), inst[j].fvalue),
            tid);
      }
    }
  }

  CHECK_EQ(pcol->Size(), Info().num_col_);

  if (sorted) {
    // sort columns
    auto ncol = static_cast<bst_omp_uint>(pcol->Size());
#pragma omp parallel for schedule(dynamic, 1) num_threads(nthread)
    for (bst_omp_uint i = 0; i < ncol; ++i) {
      if (pcol->offset[i] < pcol->offset[i + 1]) {
        std::sort(dmlc::BeginPtr(pcol->data) + pcol->offset[i],
          dmlc::BeginPtr(pcol->data) + pcol->offset[i + 1],
          Entry::CmpValue);
      }
    }
  }
}

bool SimpleDMatrix::SingleColBlock() const {
  return true;
}
}  // namespace data
}  // namespace xgboost
