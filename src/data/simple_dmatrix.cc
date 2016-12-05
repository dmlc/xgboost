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
  if (data_ptr_ >= cpages_.size()) return false;
  data_ptr_ += 1;
  SparsePage* pcol = cpages_[data_ptr_ - 1].get();
  batch_.size = col_index_.size();
  col_data_.resize(col_index_.size(), SparseBatch::Inst(NULL, 0));
  for (size_t i = 0; i < col_data_.size(); ++i) {
    const bst_uint ridx = col_index_[i];
    col_data_[i] = SparseBatch::Inst
        (dmlc::BeginPtr(pcol->data) + pcol->offset[ridx],
         static_cast<bst_uint>(pcol->offset[ridx + 1] - pcol->offset[ridx]));
  }
  batch_.col_index = dmlc::BeginPtr(col_index_);
  batch_.col_data = dmlc::BeginPtr(col_data_);
  return true;
}

dmlc::DataIter<ColBatch>* SimpleDMatrix::ColIterator() {
  size_t ncol = this->info().num_col;
  col_iter_.col_index_.resize(ncol);
  for (size_t i = 0; i < ncol; ++i) {
    col_iter_.col_index_[i] = static_cast<bst_uint>(i);
  }
  col_iter_.BeforeFirst();
  return &col_iter_;
}

dmlc::DataIter<ColBatch>* SimpleDMatrix::ColIterator(const std::vector<bst_uint>&fset) {
  size_t ncol = this->info().num_col;
  col_iter_.col_index_.resize(0);
  for (size_t i = 0; i < fset.size(); ++i) {
    if (fset[i] < ncol) col_iter_.col_index_.push_back(fset[i]);
  }
  col_iter_.BeforeFirst();
  return &col_iter_;
}

void SimpleDMatrix::InitColAccess(const std::vector<bool> &enabled,
                                  float pkeep,
                                  size_t max_row_perbatch) {
  if (this->HaveColAccess()) return;

  col_iter_.cpages_.clear();
  if (info().num_row < max_row_perbatch) {
    std::unique_ptr<SparsePage> page(new SparsePage());
    this->MakeOneBatch(enabled, pkeep, page.get());
    col_iter_.cpages_.push_back(std::move(page));
  } else {
    this->MakeManyBatch(enabled, pkeep, max_row_perbatch);
  }
  // setup col-size
  col_size_.resize(info().num_col);
  std::fill(col_size_.begin(), col_size_.end(), 0);
  for (size_t i = 0; i < col_iter_.cpages_.size(); ++i) {
    SparsePage *pcol = col_iter_.cpages_[i].get();
    for (size_t j = 0; j < pcol->Size(); ++j) {
      col_size_[j] += pcol->offset[j + 1] - pcol->offset[j];
    }
  }
}

// internal function to make one batch from row iter.
void SimpleDMatrix::MakeOneBatch(const std::vector<bool>& enabled,
                                 float pkeep,
                                 SparsePage *pcol) {
  // clear rowset
  buffered_rowset_.clear();
  // bit map
  const int nthread = omp_get_max_threads();
  std::vector<bool> bmap;
  pcol->Clear();
  common::ParallelGroupBuilder<SparseBatch::Entry>
      builder(&pcol->offset, &pcol->data);
  builder.InitBudget(info().num_col, nthread);
  // start working
  dmlc::DataIter<RowBatch>* iter = this->RowIterator();
  iter->BeforeFirst();
  while (iter->Next()) {
    const RowBatch& batch = iter->Value();
    bmap.resize(bmap.size() + batch.size, true);
    std::bernoulli_distribution coin_flip(pkeep);
    auto& rnd = common::GlobalRandom();

    long batch_size = static_cast<long>(batch.size); // NOLINT(*)
    for (long i = 0; i < batch_size; ++i) { // NOLINT(*)
      bst_uint ridx = static_cast<bst_uint>(batch.base_rowid + i);
      if (pkeep == 1.0f || coin_flip(rnd)) {
        buffered_rowset_.push_back(ridx);
      } else {
        bmap[i] = false;
      }
    }
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < batch_size; ++i) { // NOLINT(*)
      int tid = omp_get_thread_num();
      bst_uint ridx = static_cast<bst_uint>(batch.base_rowid + i);
      if (bmap[ridx]) {
        RowBatch::Inst inst = batch[i];
        for (bst_uint j = 0; j < inst.length; ++j) {
          if (enabled[inst[j].index]) {
            builder.AddBudget(inst[j].index, tid);
          }
        }
      }
    }
  }
  builder.InitStorage();

  iter->BeforeFirst();
  while (iter->Next()) {
    const RowBatch& batch = iter->Value();
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < static_cast<long>(batch.size); ++i) { // NOLINT(*)
      int tid = omp_get_thread_num();
      bst_uint ridx = static_cast<bst_uint>(batch.base_rowid + i);
      if (bmap[ridx]) {
        RowBatch::Inst inst = batch[i];
        for (bst_uint j = 0; j < inst.length; ++j) {
          if (enabled[inst[j].index]) {
            builder.Push(inst[j].index,
                         SparseBatch::Entry((bst_uint)(batch.base_rowid+i),
                                            inst[j].fvalue), tid);
          }
        }
      }
    }
  }

  CHECK_EQ(pcol->Size(), info().num_col);
  // sort columns
  bst_omp_uint ncol = static_cast<bst_omp_uint>(pcol->Size());
  #pragma omp parallel for schedule(dynamic, 1) num_threads(nthread)
  for (bst_omp_uint i = 0; i < ncol; ++i) {
    if (pcol->offset[i] < pcol->offset[i + 1]) {
      std::sort(dmlc::BeginPtr(pcol->data) + pcol->offset[i],
                dmlc::BeginPtr(pcol->data) + pcol->offset[i + 1],
                SparseBatch::Entry::CmpValue);
    }
  }
}

void SimpleDMatrix::MakeManyBatch(const std::vector<bool>& enabled,
                                  float pkeep,
                                  size_t max_row_perbatch) {
  size_t btop = 0;
  std::bernoulli_distribution coin_flip(pkeep);
  auto& rnd = common::GlobalRandom();
  buffered_rowset_.clear();
  // internal temp cache
  SparsePage tmp; tmp.Clear();
  // start working
  dmlc::DataIter<RowBatch>* iter = this->RowIterator();
  iter->BeforeFirst();

  while (iter->Next()) {
    const RowBatch &batch = iter->Value();
    for (size_t i = 0; i < batch.size; ++i) {
      bst_uint ridx = static_cast<bst_uint>(batch.base_rowid + i);
      if (pkeep == 1.0f || coin_flip(rnd)) {
        buffered_rowset_.push_back(ridx);
        tmp.Push(batch[i]);
      }
      if (tmp.Size() >= max_row_perbatch) {
        std::unique_ptr<SparsePage> page(new SparsePage());
        this->MakeColPage(tmp.GetRowBatch(0), btop, enabled, page.get());
        col_iter_.cpages_.push_back(std::move(page));
        btop = buffered_rowset_.size();
        tmp.Clear();
      }
    }
  }

  if (tmp.Size() != 0) {
    std::unique_ptr<SparsePage> page(new SparsePage());
    this->MakeColPage(tmp.GetRowBatch(0), btop, enabled, page.get());
    col_iter_.cpages_.push_back(std::move(page));
  }
}

// make column page from subset of rowbatchs
void SimpleDMatrix::MakeColPage(const RowBatch& batch,
                                size_t buffer_begin,
                                const std::vector<bool>& enabled,
                                SparsePage* pcol) {
  const int nthread = std::min(omp_get_max_threads(), std::max(omp_get_num_procs() / 2 - 2, 1));
  pcol->Clear();
  common::ParallelGroupBuilder<SparseBatch::Entry>
      builder(&pcol->offset, &pcol->data);
  builder.InitBudget(info().num_col, nthread);
  bst_omp_uint ndata = static_cast<bst_uint>(batch.size);
  #pragma omp parallel for schedule(static) num_threads(nthread)
  for (bst_omp_uint i = 0; i < ndata; ++i) {
    int tid = omp_get_thread_num();
    RowBatch::Inst inst = batch[i];
    for (bst_uint j = 0; j < inst.length; ++j) {
      const SparseBatch::Entry &e = inst[j];
      if (enabled[e.index]) {
        builder.AddBudget(e.index, tid);
      }
    }
  }
  builder.InitStorage();
  #pragma omp parallel for schedule(static) num_threads(nthread)
  for (bst_omp_uint i = 0; i < ndata; ++i) {
    int tid = omp_get_thread_num();
    RowBatch::Inst inst = batch[i];
    for (bst_uint j = 0; j < inst.length; ++j) {
      const SparseBatch::Entry &e = inst[j];
      builder.Push(
          e.index,
          SparseBatch::Entry(buffered_rowset_[i + buffer_begin], e.fvalue),
          tid);
    }
  }
  CHECK_EQ(pcol->Size(), info().num_col);
  // sort columns
  bst_omp_uint ncol = static_cast<bst_omp_uint>(pcol->Size());
  #pragma omp parallel for schedule(dynamic, 1) num_threads(nthread)
  for (bst_omp_uint i = 0; i < ncol; ++i) {
    if (pcol->offset[i] < pcol->offset[i + 1]) {
      std::sort(dmlc::BeginPtr(pcol->data) + pcol->offset[i],
                dmlc::BeginPtr(pcol->data) + pcol->offset[i + 1],
                SparseBatch::Entry::CmpValue);
    }
  }
}

bool SimpleDMatrix::SingleColBlock() const {
  return col_iter_.cpages_.size() <= 1;
}
}  // namespace data
}  // namespace xgboost
