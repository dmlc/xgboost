/*!
 * Copyright 2014 by Contributors
 * \file simple_fmatrix-inl.hpp
 * \brief the input data structure for gradient boosting
 * \author Tianqi Chen
 */
#ifndef XGBOOST_IO_SIMPLE_FMATRIX_INL_HPP_
#define XGBOOST_IO_SIMPLE_FMATRIX_INL_HPP_

#include <limits>
#include <algorithm>
#include <vector>
#include "../data.h"
#include "../utils/utils.h"
#include "../utils/random.h"
#include "../utils/omp.h"
#include "../learner/dmatrix.h"
#include "../utils/group_data.h"
#include "./sparse_batch_page.h"

namespace xgboost {
namespace io {
/*!
 * \brief sparse matrix that support column access, CSC
 */
class FMatrixS : public IFMatrix {
 public:
  typedef SparseBatch::Entry Entry;
  /*! \brief constructor */
  FMatrixS(utils::IIterator<RowBatch> *iter,
               const learner::MetaInfo &info)
      : info_(info) {
    this->iter_ = iter;
  }
  // destructor
  virtual ~FMatrixS(void) {
    if (iter_ != NULL) delete iter_;
  }
  /*! \return whether column access is enabled */
  virtual bool HaveColAccess(void) const {
    return col_size_.size() != 0;
  }
  /*! \brief get number of colmuns */
  virtual size_t NumCol(void) const {
    utils::Check(this->HaveColAccess(), "NumCol:need column access");
    return col_size_.size();
  }
  /*! \brief get number of buffered rows */
  virtual const std::vector<bst_uint> &buffered_rowset(void) const {
    return buffered_rowset_;
  }
  /*! \brief get column size */
  virtual size_t GetColSize(size_t cidx) const {
    return col_size_[cidx];
  }
  /*! \brief get column density */
  virtual float GetColDensity(size_t cidx) const {
    size_t nmiss = buffered_rowset_.size() - col_size_[cidx];
    return 1.0f - (static_cast<float>(nmiss)) / buffered_rowset_.size();
  }
  virtual void InitColAccess(const std::vector<bool> &enabled,
                             float pkeep, size_t max_row_perbatch) {
    if (this->HaveColAccess()) return;
    this->InitColData(enabled, pkeep, max_row_perbatch);
  }
  /*!
   * \brief get the row iterator associated with FMatrix
   */
  virtual utils::IIterator<RowBatch>* RowIterator(void) {
    iter_->BeforeFirst();
    return iter_;
  }
  /*!
   * \brief get the column based  iterator
   */
  virtual utils::IIterator<ColBatch>* ColIterator(void) {
    size_t ncol = this->NumCol();
    col_iter_.col_index_.resize(ncol);
    for (size_t i = 0; i < ncol; ++i) {
      col_iter_.col_index_[i] = static_cast<bst_uint>(i);
    }
    col_iter_.BeforeFirst();
    return &col_iter_;
  }
  /*!
   * \brief colmun based iterator
   */
  virtual utils::IIterator<ColBatch> *ColIterator(const std::vector<bst_uint> &fset) {
    size_t ncol = this->NumCol();
    col_iter_.col_index_.resize(0);
    for (size_t i = 0; i < fset.size(); ++i) {
      if (fset[i] < ncol) col_iter_.col_index_.push_back(fset[i]);
    }
    col_iter_.BeforeFirst();
    return &col_iter_;
  }
  /*!
   * \brief save column access data into stream
   * \param fo output stream to save to
   */
  inline void SaveColAccess(utils::IStream &fo) const { // NOLINT(*)
    size_t n = 0;
    fo.Write(&n, sizeof(n));
  }
  /*!
   * \brief load column access data from stream
   * \param fo output stream to load from
   */
  inline void LoadColAccess(utils::IStream &fi) { // NOLINT(*)
    // do nothing in load col access
  }

 protected:
  /*!
   * \brief intialize column data
   * \param enabled the list of enabled columns
   * \param pkeep probability to keep a row
   * \param max_row_perbatch maximum row per batch
   */
  inline void InitColData(const std::vector<bool> &enabled,
                          float pkeep, size_t max_row_perbatch) {
    col_iter_.Clear();
    if (info_.num_row() < max_row_perbatch) {
      SparsePage *page = new SparsePage();
      this->MakeOneBatch(enabled, pkeep, page);
      col_iter_.cpages_.push_back(page);
    } else {
      this->MakeManyBatch(enabled, pkeep, max_row_perbatch);
    }
    // setup col-size
    col_size_.resize(info_.num_col());
    std::fill(col_size_.begin(), col_size_.end(), 0);
    for (size_t i = 0; i < col_iter_.cpages_.size(); ++i) {
      SparsePage *pcol = col_iter_.cpages_[i];
      for (size_t j = 0; j < pcol->Size(); ++j) {
        col_size_[j] += pcol->offset[j + 1] - pcol->offset[j];
      }
    }
  }
  /*!
   * \brief make column page from iterator
   * \param pkeep probability to keep a row
   * \param pcol the target column
   */
  inline void MakeOneBatch(const std::vector<bool> &enabled,
                           float pkeep,
                           SparsePage *pcol) {
    // clear rowset
    buffered_rowset_.clear();
    // bit map
    int nthread;
    std::vector<bool> bmap;
    #pragma omp parallel
    {
      nthread = omp_get_num_threads();
    }
    pcol->Clear();
    utils::ParallelGroupBuilder<SparseBatch::Entry>
        builder(&pcol->offset, &pcol->data);
    builder.InitBudget(info_.num_col(), nthread);
    // start working
    iter_->BeforeFirst();
    while (iter_->Next()) {
      const RowBatch &batch = iter_->Value();
      bmap.resize(bmap.size() + batch.size, true);
      long batch_size = static_cast<long>(batch.size); // NOLINT(*)
      for (long i = 0; i < batch_size; ++i) { // NOLINT(*)
        bst_uint ridx = static_cast<bst_uint>(batch.base_rowid + i);
        if (pkeep == 1.0f || random::SampleBinary(pkeep)) {
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

    iter_->BeforeFirst();
    while (iter_->Next()) {
      const RowBatch &batch = iter_->Value();
      #pragma omp parallel for schedule(static)
      for (long i = 0; i < static_cast<long>(batch.size); ++i) { // NOLINT(*)
        int tid = omp_get_thread_num();
        bst_uint ridx = static_cast<bst_uint>(batch.base_rowid + i);
        if (bmap[ridx]) {
          RowBatch::Inst inst = batch[i];
          for (bst_uint j = 0; j < inst.length; ++j) {
            if (enabled[inst[j].index]) {
              builder.Push(inst[j].index,
                           Entry((bst_uint)(batch.base_rowid+i),
                                 inst[j].fvalue), tid);
            }
          }
        }
      }
    }

    utils::Assert(pcol->Size() == info_.num_col(),
                  "inconsistent col data");
    // sort columns
    bst_omp_uint ncol = static_cast<bst_omp_uint>(pcol->Size());
    #pragma omp parallel for schedule(dynamic, 1) num_threads(nthread)
    for (bst_omp_uint i = 0; i < ncol; ++i) {
      if (pcol->offset[i] < pcol->offset[i + 1]) {
        std::sort(BeginPtr(pcol->data) + pcol->offset[i],
                  BeginPtr(pcol->data) + pcol->offset[i + 1],
                  SparseBatch::Entry::CmpValue);
      }
    }
  }

  inline void MakeManyBatch(const std::vector<bool> &enabled,
                            float pkeep, size_t max_row_perbatch) {
    size_t btop = 0;
    buffered_rowset_.clear();
    // internal temp cache
    SparsePage tmp; tmp.Clear();
    iter_->BeforeFirst();
    while (iter_->Next()) {
      const RowBatch &batch = iter_->Value();
      for (size_t i = 0; i < batch.size; ++i) {
        bst_uint ridx = static_cast<bst_uint>(batch.base_rowid + i);
        if (pkeep == 1.0f || random::SampleBinary(pkeep)) {
          buffered_rowset_.push_back(ridx);
          tmp.Push(batch[i]);
        }
        if (tmp.Size() >= max_row_perbatch) {
          SparsePage *page = new SparsePage();
          this->MakeColPage(tmp.GetRowBatch(0),
                            BeginPtr(buffered_rowset_) + btop,
                            enabled, page);
          col_iter_.cpages_.push_back(page);
          btop = buffered_rowset_.size();
          tmp.Clear();
        }
      }
    }
    if (tmp.Size() != 0) {
      SparsePage *page = new SparsePage();
      this->MakeColPage(tmp.GetRowBatch(0),
                        BeginPtr(buffered_rowset_) + btop,
                        enabled, page);
      col_iter_.cpages_.push_back(page);
    }
  }
  // make column page from subset of rowbatchs
  inline void MakeColPage(const RowBatch &batch,
                          const bst_uint *ridx,
                          const std::vector<bool> &enabled,
                          SparsePage *pcol) {
    int nthread;
    #pragma omp parallel
    {
      nthread = omp_get_num_threads();
      int max_nthread = std::max(omp_get_num_procs() / 2 - 2, 1);
      if (nthread > max_nthread) {
        nthread = max_nthread;
      }
    }
    pcol->Clear();
    utils::ParallelGroupBuilder<SparseBatch::Entry>
        builder(&pcol->offset, &pcol->data);
    builder.InitBudget(info_.num_col(), nthread);
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
        builder.Push(e.index,
                     SparseBatch::Entry(ridx[i], e.fvalue),
                     tid);
      }
    }
    utils::Assert(pcol->Size() == info_.num_col(), "inconsistent col data");
    // sort columns
    bst_omp_uint ncol = static_cast<bst_omp_uint>(pcol->Size());
    #pragma omp parallel for schedule(dynamic, 1) num_threads(nthread)
    for (bst_omp_uint i = 0; i < ncol; ++i) {
      if (pcol->offset[i] < pcol->offset[i + 1]) {
        std::sort(BeginPtr(pcol->data) + pcol->offset[i],
                  BeginPtr(pcol->data) + pcol->offset[i + 1],
                  SparseBatch::Entry::CmpValue);
      }
    }
  }

 private:
  // one batch iterator that return content in the matrix
  struct ColBatchIter: utils::IIterator<ColBatch> {
    ColBatchIter(void) : data_ptr_(0) {}
    virtual ~ColBatchIter(void) {
      this->Clear();
    }
    virtual void BeforeFirst(void) {
      data_ptr_ = 0;
    }
    virtual bool Next(void) {
      if (data_ptr_ >= cpages_.size()) return false;
      data_ptr_ += 1;
      SparsePage *pcol = cpages_[data_ptr_ - 1];
      batch_.size = col_index_.size();
      col_data_.resize(col_index_.size(), SparseBatch::Inst(NULL, 0));
      for (size_t i = 0; i < col_data_.size(); ++i) {
        const bst_uint ridx = col_index_[i];
        col_data_[i] = SparseBatch::Inst
            (BeginPtr(pcol->data) + pcol->offset[ridx],
             static_cast<bst_uint>(pcol->offset[ridx + 1] - pcol->offset[ridx]));
      }
      batch_.col_index = BeginPtr(col_index_);
      batch_.col_data = BeginPtr(col_data_);
      return true;
    }
    virtual const ColBatch &Value(void) const {
      return batch_;
    }
    inline void Clear(void) {
      for (size_t i = 0; i < cpages_.size(); ++i) {
        delete cpages_[i];
      }
      cpages_.clear();
    }
    // data content
    std::vector<bst_uint> col_index_;
    // column content
    std::vector<ColBatch::Inst> col_data_;
    // column sparse pages
    std::vector<SparsePage*> cpages_;
    // data pointer
    size_t data_ptr_;
    // temporal space for batch
    ColBatch batch_;
  };
  // --- data structure used to support InitColAccess --
  // column iterator
  ColBatchIter col_iter_;
  // shared meta info with DMatrix
  const learner::MetaInfo &info_;
  // row iterator
  utils::IIterator<RowBatch> *iter_;
  /*! \brief list of row index that are buffered */
  std::vector<bst_uint> buffered_rowset_;
  // count for column data
  std::vector<size_t> col_size_;
};
}  // namespace io
}  // namespace xgboost
#endif  // XGBOOST_IO_SLICE_FMATRIX_INL_HPP_
