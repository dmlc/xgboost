#ifndef XGBOOST_IO_PAGE_FMATRIX_INL_HPP_
#define XGBOOST_IO_PAGE_FMATRIX_INL_HPP_
/*!
 * \file page_fmatrix-inl.hpp
 * sparse page manager for fmatrix
 * \author Tianqi Chen
 */
#include <vector>
#include <string>
#include <algorithm>
#include "../data.h"
#include "../utils/iterator.h"
#include "../utils/io.h"
#include "../utils/matrix_csr.h"
#include "../utils/thread_buffer.h"
namespace xgboost {
namespace io {
class CSCMatrixManager {
 public:
  /*! \brief in memory page */
  struct Page {
   public:
    /*! \brief initialize the page */
    explicit Page(size_t size) {
      buffer.resize(size);
      col_index.reserve(10);
      col_data.reserve(10);
    }
    /*! \brief clear the page */
    inline void Clear(void) {
      num_entry = 0;
      col_index.clear();
      col_data.clear();
    }
    /*! \brief number of used entries */
    size_t num_entry;
    /*! \brief column index */
    std::vector<bst_uint> col_index;
    /*! \brief column data */
    std::vector<ColBatch::Inst> col_data;
    /*! \brief number of free entries */
    inline size_t NumFreeEntry(void) const {
      return buffer.size() - num_entry;
    }
    inline ColBatch::Entry* AllocEntry(size_t len) {
      ColBatch::Entry *p_data = &buffer[0] + num_entry;
      num_entry += len;
      return p_data;
    }
    /*! \brief get underlying batch */
    inline ColBatch GetBatch(void) const {
      ColBatch batch;
      batch.size = col_index.size();
      batch.col_index = BeginPtr(col_index);
      batch.col_data  = BeginPtr(col_data);
      return batch;
    }

   private:
    /*! \brief buffer space, not to be changed since ready */
    std::vector<ColBatch::Entry> buffer;
  };
  /*! \brief define type of page pointer */
  typedef Page *PagePtr;
  // constructor
  CSCMatrixManager(void) {
    fi_ = NULL;
  }
  /*! \brief get column pointer */
  inline const std::vector<size_t> &col_ptr(void) const {
    return col_ptr_;
  }
  inline void SetParam(const char *name, const char *val) {
  }
  inline PagePtr Create(void) {
    return new Page(page_size_);
  }
  inline void FreeSpace(PagePtr &a) {
    delete a;
  }
  inline void Destroy(void) {
  }
  inline void BeforeFirst(void) {
    col_index_ = col_todo_;
    read_top_ = 0;
  }
  inline bool LoadNext(PagePtr &val) {
    val->Clear();
    if (read_top_ >= col_index_.size()) return false;
    while (read_top_ < col_index_.size()) {
      if (!this->TryFill(col_index_[read_top_], val)) {
        return true;
      }
      ++read_top_;
    }
    return true;
  }
  inline bool Init(void) {
    this->BeforeFirst();
    return true;
  }
  inline void Setup(utils::ISeekStream *fi, double page_ratio) {
    fi_ = fi;
    fi_->Read(&begin_meta_ , sizeof(begin_meta_));
    begin_data_ = static_cast<size_t>(fi->Tell());
    fi_->Seek(begin_meta_);
    fi_->Read(&col_ptr_);
    size_t psmax = 0;
    for (size_t i = 0; i < col_ptr_.size() - 1; ++i) {
      psmax = std::max(psmax, col_ptr_[i+1] - col_ptr_[i]);
    }
    utils::Check(page_ratio >= 1.0f, "col_page_ratio must be at least 1");
    page_size_ = std::max(static_cast<size_t>(psmax * page_ratio), psmax);
  }
  inline void SetColSet(const std::vector<bst_uint> &cset, bool setall) {
    if (!setall) {
      col_todo_.resize(0);
      for (size_t i = 0; i < cset.size(); ++i) {
        if (col_todo_[i] < static_cast<bst_uint>(col_ptr_.size() - 1)) {
          col_todo_.push_back(cset[i]);
        }
      }
      std::sort(col_todo_.begin(), col_todo_.end());
    } else {
      col_todo_.resize(col_ptr_.size()-1);
      for (size_t i = 0; i < col_todo_.size(); ++i) {
        col_todo_[i] = static_cast<bst_uint>(i);
      }
    }
  }

 private:
  /*! \brief fill a page with */
  inline bool TryFill(size_t cidx, Page *p_page) {
    size_t len = col_ptr_[cidx+1] - col_ptr_[cidx];
    if (p_page->NumFreeEntry() < len) return false;
    ColBatch::Entry *p_data = p_page->AllocEntry(len);
    fi_->Seek(col_ptr_[cidx] * sizeof(ColBatch::Entry) + begin_data_);
    utils::Check(fi_->Read(p_data, sizeof(ColBatch::Entry) * len) != 0,
                 "invalid column buffer format");
    p_page->col_data.push_back(ColBatch::Inst(p_data, static_cast<bst_uint>(len)));
    p_page->col_index.push_back(static_cast<bst_uint>(cidx));
    return true;
  }
  // the following are in memory auxiliary data structure
  /*! \brief top of reader position */
  size_t read_top_;
  /*! \brief size of page */
  size_t page_size_;
  /*! \brief column index to be loaded */
  std::vector<bst_uint> col_index_;
  /*! \brief column index to be after calling before first */
  std::vector<bst_uint> col_todo_;
  // the following are input content
  /*! \brief beginning position of data content */
  size_t begin_data_;
  /*! \brief size of data content */
  size_t begin_meta_;
  /*! \brief input stream */
  utils::ISeekStream *fi_;
  /*! \brief column pointer of CSC format */
  std::vector<size_t> col_ptr_;
};

class ThreadColPageIterator : public utils::IIterator<ColBatch> {
 public:
  explicit ThreadColPageIterator(utils::ISeekStream *fi,
                                 float page_ratio, bool silent) {
    itr_.SetParam("buffer_size", "2");
    itr_.get_factory().Setup(fi, page_ratio);
    itr_.Init();
    if (!silent) {
      utils::Printf("ThreadColPageIterator: finish initialzing, %u columns\n",
                    static_cast<unsigned>(col_ptr().size() - 1));
    }
  }
  virtual ~ThreadColPageIterator(void) {
  }
  virtual void BeforeFirst(void) {
    itr_.BeforeFirst();
  }
  virtual bool Next(void) {
    // page to be loaded
    CSCMatrixManager::PagePtr page;
    if (!itr_.Next(page)) return false;
    out_ = page->GetBatch();
    return true;
  }
  virtual const ColBatch &Value(void) const {
    return out_;
  }
  inline const std::vector<size_t> &col_ptr(void) const {
    return itr_.get_factory().col_ptr();
  }
  inline void SetColSet(const std::vector<bst_uint> &cset,
                        bool setall = false) {
    itr_.get_factory().SetColSet(cset, setall);
  }

 private:
  // output data
  ColBatch out_;
  // internal iterator
  utils::ThreadBuffer<CSCMatrixManager::PagePtr, CSCMatrixManager> itr_;
};
/*!
 * \brief sparse matrix that support column access
 */
class FMatrixPage : public IFMatrix {
 public:
  /*! \brief constructor */
  FMatrixPage(utils::IIterator<RowBatch> *iter, std::string fname_buffer)
      : fname_cbuffer_(fname_buffer) {
    this->row_iter_ = iter;
    this->col_iter_ = NULL;
    this->fi_ = NULL;
  }
  // destructor
  virtual ~FMatrixPage(void) {
    if (row_iter_ != NULL) delete row_iter_;
    if (col_iter_ != NULL) delete col_iter_;
    if (fi_ != NULL) {
      fi_->Close(); delete fi_;
    }
  }
  /*! \return whether column access is enabled */
  virtual bool HaveColAccess(void) const {
    return col_iter_ != NULL;
  }
  /*! \brief get number of colmuns */
  virtual size_t NumCol(void) const {
    utils::Check(this->HaveColAccess(), "NumCol:need column access");
    return col_iter_->col_ptr().size() - 1;
  }
  /*! \brief get number of buffered rows */
  virtual const std::vector<bst_uint> &buffered_rowset(void) const {
    return buffered_rowset_;
  }
  /*! \brief get column size */
  virtual size_t GetColSize(size_t cidx) const {
    const std::vector<size_t> &col_ptr = col_iter_->col_ptr();
    return col_ptr[cidx+1] - col_ptr[cidx];
  }
  /*! \brief get column density */
  virtual float GetColDensity(size_t cidx) const {
    const std::vector<size_t> &col_ptr = col_iter_->col_ptr();
    size_t nmiss = buffered_rowset_.size() - (col_ptr[cidx+1] - col_ptr[cidx]);
    return 1.0f - (static_cast<float>(nmiss)) / buffered_rowset_.size();
  }
  virtual void InitColAccess(const std::vector<bool> &enabled, float pkeep = 1.0f) {
    if (this->HaveColAccess()) return;
    utils::Printf("start to initialize page col access\n");
    if (this->LoadColData()) {
      utils::Printf("loading previously saved col data\n");
      return;
    }
    this->InitColData(pkeep, fname_cbuffer_.c_str(),
                      1 << 30, 5);
    utils::Check(this->LoadColData(), "fail to read in column data");
    utils::Printf("finish initialize page col access\n");
  }
  /*!
   * \brief get the row iterator associated with FMatrix
   */
  virtual utils::IIterator<RowBatch>* RowIterator(void) {
    row_iter_->BeforeFirst();
    return row_iter_;
  }
  /*!
   * \brief get the column based  iterator
   */
  virtual utils::IIterator<ColBatch>* ColIterator(void) {
    std::vector<bst_uint> cset;
    col_iter_->SetColSet(cset, true);
    col_iter_->BeforeFirst();
    return col_iter_;
  }
  /*!
   * \brief colmun based iterator
   */
  virtual utils::IIterator<ColBatch> *ColIterator(const std::vector<bst_uint> &fset) {
    col_iter_->SetColSet(fset, false);
    col_iter_->BeforeFirst();
    return col_iter_;
  }

 protected:
  /*!
   * \brief try load column data from file
   */
  inline bool LoadColData(void) {
    FILE *fp = fopen64(fname_cbuffer_.c_str(), "rb");
    if (fp == NULL) return false;
    fi_ = new utils::FileStream(fp);
    static_cast<utils::IStream*>(fi_)->Read(&buffered_rowset_);
    col_iter_ = new ThreadColPageIterator(fi_, 2.0f, false);
    return true;
  }
  /*!
   * \brief intialize column data
   * \param pkeep probability to keep a row
   */
  inline void InitColData(float pkeep, const char *fname,
                          size_t buffer_size, size_t col_step) {
    buffered_rowset_.clear();
    utils::FileStream fo(utils::FopenCheck(fname, "wb+"));
    // use 64M buffer
    utils::SparseCSRFileBuilder<ColBatch::Entry> builder(&fo, buffer_size);
    // start working
    row_iter_->BeforeFirst();
    while (row_iter_->Next()) {
      const RowBatch &batch = row_iter_->Value();
      for (size_t i = 0; i < batch.size; ++i) {
        if (pkeep == 1.0f || random::SampleBinary(pkeep)) {
          buffered_rowset_.push_back(static_cast<bst_uint>(batch.base_rowid+i));
          RowBatch::Inst inst = batch[i];
          for (bst_uint j = 0; j < inst.length; ++j) {
            builder.AddBudget(inst[j].index);
          }
        }
      }
    }
    // write buffered rowset
    static_cast<utils::IStream*>(&fo)->Write(buffered_rowset_);
    builder.InitStorage();
    row_iter_->BeforeFirst();
    size_t ktop = 0;
    while (row_iter_->Next()) {
      const RowBatch &batch = row_iter_->Value();
      for (size_t i = 0; i < batch.size; ++i) {
        if (ktop < buffered_rowset_.size() &&
            buffered_rowset_[ktop] == batch.base_rowid + i) {
          ++ktop;
          RowBatch::Inst inst = batch[i];
          for (bst_uint j = 0; j < inst.length; ++j) {
            builder.PushElem(inst[j].index,
                             ColBatch::Entry((bst_uint)(batch.base_rowid+i),
                                             inst[j].fvalue));
          }
          if (ktop % 100000 == 0) {
            utils::Printf("\r                         \r");
            utils::Printf("InitCol: %lu rows ", static_cast<unsigned long>(ktop));    
          }
        }
      }
    }
    builder.Finalize();
    builder.SortRows(ColBatch::Entry::CmpValue, col_step);
    fo.Close();
  }

 private:
  // row iterator
  utils::IIterator<RowBatch> *row_iter_;
  // column iterator
  ThreadColPageIterator *col_iter_;
  // file pointer to data
  utils::FileStream *fi_;
  // file name of column buffer
  std::string fname_cbuffer_;
  /*! \brief list of row index that are buffered */
  std::vector<bst_uint> buffered_rowset_;
};

class DMatrixColPage : public DMatrixPageBase<0xffffab03> {
 public:
  explicit DMatrixColPage(const char *fname) {
    fmat_ = new FMatrixPage(iter_, fname);
  }
  virtual ~DMatrixColPage(void) {
    delete fmat_;
  }
  virtual IFMatrix *fmat(void) const {
    return fmat_;
  }
  /*! \brief the real fmatrix */
  IFMatrix *fmat_;
};

}  // namespace io
}  // namespace xgboost
#endif  // XGBOOST_IO_PAGE_FMATRIX_INL_HPP_
