/*!
 * Copyright (c) 2014 by Contributors
 * \file page_fmatrix-inl.hpp
 *   col iterator based on sparse page
 * \author Tianqi Chen
 */
#ifndef XGBOOST_IO_PAGE_FMATRIX_INL_HPP_
#define XGBOOST_IO_PAGE_FMATRIX_INL_HPP_

#include <vector>
#include <string>
#include <algorithm>

namespace xgboost {
namespace io {
/*! \brief thread buffer iterator */
class ThreadColPageIterator: public utils::IIterator<ColBatch> {
 public:
  ThreadColPageIterator(void) {
    itr.SetParam("buffer_size", "2");
    page_ = NULL;
  }
  virtual ~ThreadColPageIterator(void) {}
  virtual void Init(void) {}
  virtual void BeforeFirst(void) {
    itr.BeforeFirst();
  }
  virtual bool Next(void) {
    if (!itr.Next(page_)) return false;
    out_.col_index = BeginPtr(itr.get_factory().index_set());
    col_data_.resize(page_->offset.size() - 1, SparseBatch::Inst(NULL, 0));
    for (size_t i = 0; i < col_data_.size(); ++i) {
      col_data_[i] = SparseBatch::Inst
          (BeginPtr(page_->data) + page_->offset[i],
           static_cast<bst_uint>(page_->offset[i + 1] - page_->offset[i]));
    }
    out_.col_data = BeginPtr(col_data_);
    out_.size = col_data_.size();
    return true;
  }
  virtual const ColBatch &Value(void) const {
    return out_;
  }
  /*! \brief load and initialize the iterator with fi */
  inline void SetFile(const utils::FileStream &fi) {
    itr.get_factory().SetFile(fi);
    itr.Init();
  }
  // set index set
  inline void SetIndexSet(const std::vector<bst_uint> &fset, bool load_all) {
    itr.get_factory().SetIndexSet(fset, load_all);
  }

 private:
  // output data
  ColBatch out_;
  SparsePage *page_;
  std::vector<SparseBatch::Inst> col_data_;
  utils::ThreadBuffer<SparsePage*, SparsePageFactory> itr;
};

struct ColConvertFactory {
  inline bool Init(void) {
    return true;
  }
  inline void Setup(float pkeep,
                    size_t max_row_perbatch,
                    size_t num_col,
                    utils::IIterator<RowBatch> *iter,
                    std::vector<bst_uint> *buffered_rowset,
                    const std::vector<bool> *enabled) {
    pkeep_ = pkeep;
    max_row_perbatch_ = max_row_perbatch;
    num_col_ = num_col;
    iter_ = iter;
    buffered_rowset_ = buffered_rowset;
    enabled_ = enabled;
  }
  inline SparsePage *Create(void) {
    return new SparsePage();
  }
  inline void FreeSpace(SparsePage *a) {
    delete a;
  }
  inline void SetParam(const char *name, const char *val) {}
  inline bool LoadNext(SparsePage *val) {
    tmp_.Clear();
    size_t btop = buffered_rowset_->size();
    while (iter_->Next()) {
      const RowBatch &batch = iter_->Value();
      for (size_t i = 0; i < batch.size; ++i) {
        bst_uint ridx = static_cast<bst_uint>(batch.base_rowid + i);
        if (pkeep_ == 1.0f || random::SampleBinary(pkeep_)) {
          buffered_rowset_->push_back(ridx);
          tmp_.Push(batch[i]);
        }
      }
      if (tmp_.MemCostBytes() >= kPageSize ||
          tmp_.Size() >= max_row_perbatch_) {
        this->MakeColPage(tmp_, BeginPtr(*buffered_rowset_) + btop,
                          *enabled_, val);
        return true;
      }
    }
    if (tmp_.Size() != 0) {
        this->MakeColPage(tmp_, BeginPtr(*buffered_rowset_) + btop,
                          *enabled_, val);
        return true;
    } else {
      return false;
    }
  }
  inline void Destroy(void) {}
  inline void BeforeFirst(void) {}
  inline void MakeColPage(const SparsePage &prow,
                          const bst_uint *ridx,
                          const std::vector<bool> &enabled,
                          SparsePage *pcol) {
    pcol->Clear();
    int nthread;
    #pragma omp parallel
    {
      nthread = omp_get_num_threads();
      int max_nthread = std::max(omp_get_num_procs() / 2 - 4, 1);
      if (nthread > max_nthread) {
        nthread = max_nthread;
      }
    }
    pcol->Clear();
    utils::ParallelGroupBuilder<SparseBatch::Entry>
        builder(&pcol->offset, &pcol->data);
    builder.InitBudget(num_col_, nthread);
    bst_omp_uint ndata = static_cast<bst_uint>(prow.Size());
    #pragma omp parallel for schedule(static) num_threads(nthread)
    for (bst_omp_uint i = 0; i < ndata; ++i) {
      int tid = omp_get_thread_num();
      for (size_t j = prow.offset[i]; j < prow.offset[i+1]; ++j) {
        const SparseBatch::Entry &e = prow.data[j];
        if (enabled[e.index]) {
          builder.AddBudget(e.index, tid);
        }
      }
    }
    builder.InitStorage();
    #pragma omp parallel for schedule(static) num_threads(nthread)
    for (bst_omp_uint i = 0; i < ndata; ++i) {
      int tid = omp_get_thread_num();
      for (size_t j = prow.offset[i]; j < prow.offset[i+1]; ++j) {
        const SparseBatch::Entry &e = prow.data[j];
        builder.Push(e.index,
                     SparseBatch::Entry(ridx[i], e.fvalue),
                     tid);
      }
    }
    utils::Assert(pcol->Size() == num_col_, "inconsistent col data");
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
  // probability of keep
  float pkeep_;
  // maximum number of rows per batch
  size_t max_row_perbatch_;
  // number of columns
  size_t num_col_;
  // row batch iterator
  utils::IIterator<RowBatch> *iter_;
  // buffered rowset
  std::vector<bst_uint> *buffered_rowset_;
  // enabled marks
  const std::vector<bool> *enabled_;
  // internal temp cache
  SparsePage tmp_;
  /*! \brief page size 256 M */
  static const size_t kPageSize = 256 << 20UL;
};
/*!
 * \brief sparse matrix that support column access, CSC
 */
class FMatrixPage : public IFMatrix {
 public:
  typedef SparseBatch::Entry Entry;
  /*! \brief constructor */
  FMatrixPage(utils::IIterator<RowBatch> *iter,
              const learner::MetaInfo &info) : info(info) {
    this->iter_ = iter;
  }
  // destructor
  virtual ~FMatrixPage(void) {
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
    size_t nmiss = num_buffered_row_ - (col_size_[cidx]);
    return 1.0f - (static_cast<float>(nmiss)) / num_buffered_row_;
  }
  virtual void InitColAccess(const std::vector<bool> &enabled,
                             float pkeep, size_t max_row_perbatch) {
    if (this->HaveColAccess()) return;
    if (TryLoadColData()) return;
    this->InitColData(enabled, pkeep, max_row_perbatch);
    utils::Check(TryLoadColData(), "failed on creating col.blob");
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
    col_index_.resize(0);
    for (size_t i = 0; i < ncol; ++i) {
      col_index_.push_back(static_cast<bst_uint>(i));
    }
    col_iter_.SetIndexSet(col_index_, false);
    col_iter_.BeforeFirst();
    return &col_iter_;
  }
  /*!
   * \brief colmun based iterator
   */
  virtual utils::IIterator<ColBatch> *ColIterator(const std::vector<bst_uint> &fset) {
    size_t ncol = this->NumCol();
    col_index_.resize(0);
    for (size_t i = 0; i < fset.size(); ++i) {
      if (fset[i] < ncol) col_index_.push_back(fset[i]);
    }
    col_iter_.SetIndexSet(col_index_, false);
    col_iter_.BeforeFirst();
    return &col_iter_;
  }
  // set the cache file name
  inline void set_cache_file(const std::string &cache_file) {
    col_data_name_ = std::string(cache_file) + ".col.blob";
    col_meta_name_ = std::string(cache_file) + ".col.meta";
  }

 protected:
  inline bool TryLoadColData(void) {
    std::FILE *fi = fopen64(col_meta_name_.c_str(), "rb");
    if (fi == NULL) return false;
    utils::FileStream fs(fi);
    LoadMeta(&fs);
    fs.Close();
    fi = utils::FopenCheck(col_data_name_.c_str(), "rb");
    if (fi == NULL) return false;
    col_iter_.SetFile(utils::FileStream(fi));
    return true;
  }
  inline void LoadMeta(utils::IStream *fi) {
    utils::Check(fi->Read(&num_buffered_row_, sizeof(num_buffered_row_)) != 0,
                 "invalid col.blob file");
    utils::Check(fi->Read(&buffered_rowset_),
                 "invalid col.blob file");
    utils::Check(fi->Read(&col_size_),
                 "invalid col.blob file");
  }
  inline void SaveMeta(utils::IStream *fo) {
    fo->Write(&num_buffered_row_, sizeof(num_buffered_row_));
    fo->Write(buffered_rowset_);
    fo->Write(col_size_);
  }
  /*!
   * \brief intialize column data
   * \param pkeep probability to keep a row
   */
  inline void InitColData(const std::vector<bool> &enabled,
                          float pkeep, size_t max_row_perbatch) {
    // clear rowset
    buffered_rowset_.clear();
    col_size_.resize(info.num_col());
    std::fill(col_size_.begin(), col_size_.end(), 0);
    utils::FileStream fo;
    fo = utils::FileStream(utils::FopenCheck(col_data_name_.c_str(), "wb"));
    iter_->BeforeFirst();
    double tstart = rabit::utils::GetTime();
    size_t bytes_write = 0;
    utils::ThreadBuffer<SparsePage*, ColConvertFactory> citer;
    citer.SetParam("buffer_size", "2");
    citer.get_factory().Setup(pkeep, max_row_perbatch, info.num_col(),
                              iter_, &buffered_rowset_, &enabled);
    citer.Init();
    SparsePage *pcol;
    while (citer.Next(pcol)) {
      for (size_t i = 0; i < pcol->Size(); ++i) {
        col_size_[i] += pcol->offset[i + 1] - pcol->offset[i];
      }
      pcol->Save(&fo);
      size_t spage = pcol->MemCostBytes();
      bytes_write += spage;
      double tnow = rabit::utils::GetTime();
      double tdiff = tnow - tstart;
      utils::Printf("Writting to %s in %g MB/s, %lu MB written\n",
                    col_data_name_.c_str(),
                    (bytes_write >> 20UL) / tdiff,
                    (bytes_write >> 20UL));
    }
    fo.Close();
    num_buffered_row_ = buffered_rowset_.size();
    fo = utils::FileStream(utils::FopenCheck(col_meta_name_.c_str(), "wb"));
    this->SaveMeta(&fo);
    fo.Close();
  }

 private:
  /*! \brief page size 256 M */
  static const size_t kPageSize = 256 << 20UL;
  // shared meta info with DMatrix
  const learner::MetaInfo &info;
  // row iterator
  utils::IIterator<RowBatch> *iter_;
  /*! \brief column based data file name */
  std::string col_data_name_;
  /*! \brief column based data file name */
  std::string col_meta_name_;
  /*! \brief list of row index that are buffered */
  std::vector<bst_uint> buffered_rowset_;
  // number of buffered rows
  size_t num_buffered_row_;
  // count for column data
  std::vector<size_t> col_size_;
  // internal column index for output
  std::vector<bst_uint> col_index_;
  // internal thread backed col iterator
  ThreadColPageIterator col_iter_;
};
}  // namespace io
}  // namespace xgboost
#endif  // XGBOOST_IO_PAGE_FMATRIX_INL_HPP_
