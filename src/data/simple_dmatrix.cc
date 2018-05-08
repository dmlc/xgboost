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
  if (data_ >= cpages_.size()) return false;
  data_ += 1;
  return true;
}

  dmlc::DataIter<SparsePage>* SimpleDMatrix::ColIterator() {
  col_iter_.BeforeFirst();
  return &col_iter_;
}

void SimpleDMatrix::InitColAccess(const std::vector<bool> &enabled,
                                  float pkeep,
                                  size_t max_row_perbatch, bool sorted) {
  if (this->HaveColAccess(sorted)) return;
  col_iter_.sorted_ = sorted;
  col_iter_.cpages_.clear();
  if (Info().num_row_ < max_row_perbatch) {
    std::unique_ptr<SparsePage> page(new SparsePage());
    this->MakeOneBatch(enabled, pkeep, page.get(), sorted);
    col_iter_.cpages_.push_back(std::move(page));
  } else {
    this->MakeManyBatch(enabled, pkeep, max_row_perbatch, sorted);
  }
  // setup col-size
  col_size_.resize(Info().num_col_);
  std::fill(col_size_.begin(), col_size_.end(), 0);
  for (auto & cpage : col_iter_.cpages_) {
    SparsePage *pcol = cpage.get();
    for (size_t j = 0; j < pcol->Size(); ++j) {
      col_size_[j] += pcol->offset[j + 1] - pcol->offset[j];
    }
  }
}

// internal function to make one batch from row iter.
void SimpleDMatrix::MakeOneBatch(const std::vector<bool>& enabled, float pkeep,
                                 SparsePage* pcol, bool sorted) {
  // clear rowset
  buffered_rowset_.Clear();
  // bit map
  const int nthread = omp_get_max_threads();
  std::vector<bool> bmap;
  pcol->Clear();
  common::ParallelGroupBuilder<Entry>
      builder(&pcol->offset, &pcol->data);
  builder.InitBudget(Info().num_col_, nthread);
  // start working
  auto iter = this->RowIterator();
  iter->BeforeFirst();
  while (iter->Next()) {
    const  auto& batch = iter->Value();
    bmap.resize(bmap.size() + batch.Size(), true);
    std::bernoulli_distribution coin_flip(pkeep);
    auto& rnd = common::GlobalRandom();

    long batch_size = static_cast<long>(batch.Size()); // NOLINT(*)
    for (long i = 0; i < batch_size; ++i) { // NOLINT(*)
      auto ridx = static_cast<bst_uint>(batch.base_rowid + i);
      if (pkeep == 1.0f || coin_flip(rnd)) {
        buffered_rowset_.PushBack(ridx);
      } else {
        bmap[i] = false;
      }
    }
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < batch_size; ++i) { // NOLINT(*)
      int tid = omp_get_thread_num();
      auto ridx = static_cast<bst_uint>(batch.base_rowid + i);
      if (bmap[ridx]) {
         auto inst = batch[i];
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
     auto batch = iter->Value();
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < static_cast<long>(batch.Size()); ++i) { // NOLINT(*)
      int tid = omp_get_thread_num();
      auto ridx = static_cast<bst_uint>(batch.base_rowid + i);
      if (bmap[ridx]) {
         auto inst = batch[i];
        for (bst_uint j = 0; j < inst.length; ++j) {
          if (enabled[inst[j].index]) {
            builder.Push(inst[j].index,
                         Entry(static_cast<bst_uint>(batch.base_rowid+i),
                                            inst[j].fvalue), tid);
          }
        }
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

void SimpleDMatrix::MakeManyBatch(const std::vector<bool>& enabled,
                                  float pkeep,
                                  size_t max_row_perbatch, bool sorted) {
  size_t btop = 0;
  std::bernoulli_distribution coin_flip(pkeep);
  auto& rnd = common::GlobalRandom();
  buffered_rowset_.Clear();
  // internal temp cache
  SparsePage tmp; tmp.Clear();
  // start working
  auto iter = this->RowIterator();
  iter->BeforeFirst();

  while (iter->Next()) {
    auto batch = iter->Value();
    for (size_t i = 0; i < batch.Size(); ++i) {
      auto ridx = static_cast<bst_uint>(batch.base_rowid + i);
      if (pkeep == 1.0f || coin_flip(rnd)) {
        buffered_rowset_.PushBack(ridx);
        tmp.Push(batch[i]);
      }
      if (tmp.Size() >= max_row_perbatch) {
        std::unique_ptr<SparsePage> page(new SparsePage());
        this->MakeColPage(tmp, btop, enabled, page.get(), sorted);
        col_iter_.cpages_.push_back(std::move(page));
        btop = buffered_rowset_.Size();
        tmp.Clear();
      }
    }
  }

  if (tmp.Size() != 0) {
    std::unique_ptr<SparsePage> page(new SparsePage());
    this->MakeColPage(tmp, btop, enabled, page.get(), sorted);
    col_iter_.cpages_.push_back(std::move(page));
  }
}

// make column page from subset of rowbatchs
void SimpleDMatrix::MakeColPage(const SparsePage& batch,
                                size_t buffer_begin,
                                const std::vector<bool>& enabled,
                                SparsePage* pcol, bool sorted) {
  const int nthread = std::min(omp_get_max_threads(), std::max(omp_get_num_procs() / 2 - 2, 1));
  pcol->Clear();
  common::ParallelGroupBuilder<Entry>
      builder(&pcol->offset, &pcol->data);
  builder.InitBudget(Info().num_col_, nthread);
  bst_omp_uint ndata = static_cast<bst_uint>(batch.Size());
  #pragma omp parallel for schedule(static) num_threads(nthread)
  for (bst_omp_uint i = 0; i < ndata; ++i) {
    int tid = omp_get_thread_num();
    SparsePage::Inst inst = batch[i];
    for (bst_uint j = 0; j < inst.length; ++j) {
      const Entry &e = inst[j];
      if (enabled[e.index]) {
        builder.AddBudget(e.index, tid);
      }
    }
  }
  builder.InitStorage();
  #pragma omp parallel for schedule(static) num_threads(nthread)
  for (bst_omp_uint i = 0; i < ndata; ++i) {
    int tid = omp_get_thread_num();
    SparsePage::Inst inst = batch[i];
    for (bst_uint j = 0; j < inst.length; ++j) {
      const Entry &e = inst[j];
      builder.Push(
          e.index,
          Entry(buffered_rowset_[i + buffer_begin], e.fvalue),
          tid);
    }
  }
  CHECK_EQ(pcol->Size(), Info().num_col_);
  // sort columns
  if (sorted) {
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
  return col_iter_.cpages_.size() <= 1;
}
}  // namespace data
}  // namespace xgboost
