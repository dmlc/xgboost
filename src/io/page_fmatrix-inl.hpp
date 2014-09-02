#ifndef XGBOOST_IO_PAGE_FMATRIX_INL_HPP_
#define XGBOOST_IO_PAGE_FMATRIX_INL_HPP_
/*!
 * \file page_fmatrix-inl.hpp
 * sparse page manager for fmatrix
 * \author Tianqi Chen
 */
#include "../data.h"
#include "../utils/iterator.h"
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
      batch.col_index = &col_index[0];
      batch.col_data  = &col_data[0];
      return batch;
    }
   private:
    /*! \brief buffer space, not to be changed since ready */
    std::vector<ColBatch::Entry> buffer;
  };
  /*! \brief define type of page pointer */
  typedef Page *PagePtr;
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
      if (!this->TryFill(col_index_[read_top_], val)) return true;
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
    fi_->Read(&begin_meta_ , sizeof(size_t));
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
      col_todo_.resize(cset.size());
      for (size_t i = 0; i < cset.size(); ++i) {
        col_todo_[i] = cset[i];
        utils::Assert(col_todo_[i] < static_cast<bst_uint>(col_ptr_.size() - 1),
                      "CSCMatrixManager: column index exceed bound");
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
    fi_->Seek(col_ptr_[cidx] * sizeof(ColBatch::Entry) + sizeof(size_t));
    utils::Check(fi_->Read(p_data, sizeof(ColBatch::Entry) * len) != 0,
                 "invalid column buffer format");
    p_page->col_data.push_back(ColBatch::Inst(p_data, len));
    p_page->col_index.push_back(cidx);
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
  /*! \brief size of data content */
  size_t begin_meta_;
  /*! \brief input stream */
  utils::ISeekStream *fi_;
  /*! \brief column pointer of CSC format */
  std::vector<size_t> col_ptr_;
};

class ThreadColPageIterator : public utils::IIterator<ColBatch> {
 public:
  ThreadColPageIterator(void) {
    itr_.SetParam("buffer_size", "2");
    page_ = NULL;
    fi_ = NULL;
    silent = 0;
  }
  virtual ~ThreadColPageIterator(void) {
    if (fi_ != NULL) {
      fi_->Close(); delete fi_;
    }
  }
  virtual void Init(void) {
    fi_ = new utils::FileStream(utils::FopenCheck(col_pagefile_.c_str(), "rb"));
    itr_.get_factory().Setup(fi_, col_pageratio_);
    if (silent == 0) {
      printf("ThreadColPageIterator: finish initialzing from %s, %u columns\n",
             col_pagefile_.c_str(), static_cast<unsigned>(col_ptr().size() - 1));
    }
  }
  virtual void SetParam(const char *name, const char *val) {
    if (!strcmp("col_pageratio", val)) col_pageratio_ = atof(val);
    if (!strcmp("col_pagefile", val)) col_pagefile_ = val;
    if (!strcmp("silent", val)) silent = atoi(val);
  }
  virtual void BeforeFirst(void) {
    itr_.BeforeFirst();
  } 
  virtual bool Next(void) {
    if(!itr_.Next(page_)) return false;
    out_ = page_->GetBatch();
    return true;
  }
  virtual const ColBatch &Value(void) const{
    return out_;
  }
  inline const std::vector<size_t> &col_ptr(void) const {
    return itr_.get_factory().col_ptr();
  }
  inline void SetColSet(const std::vector<bst_uint> &cset, bool setall = false) {
    itr_.get_factory().SetColSet(cset, setall);
  }

 private:
  // shutup
  int silent;
  // input file
  utils::FileStream *fi_;
  // size of page
  float col_pageratio_;
  // name of file
  std::string col_pagefile_;
  // output data
  ColBatch out_;
  // page to be loaded
  CSCMatrixManager::PagePtr page_;
  // internal iterator
  utils::ThreadBuffer<CSCMatrixManager::PagePtr,CSCMatrixManager> itr_;
};

/*!
 * \brief sparse matrix that support column access
 */
class FMatrixPage : public IFMatrix {
 public:
  /*! \brief constructor */
  FMatrixPage(utils::IIterator<RowBatch> *iter) {
    this->row_iter_ = iter;
    this->col_iter_ = NULL;
  }
  // destructor
  virtual ~FMatrixPage(void) {
    if (row_iter_ != NULL) delete row_iter_;
    if (col_iter_ != NULL) delete col_iter_;
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
  virtual void InitColAccess(float pkeep = 1.0f) {
    if (this->HaveColAccess()) return;
    this->InitColData(pkeep);
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
   * \brief intialize column data
   * \param pkeep probability to keep a row
   */
  inline void InitColData(float pkeep) {
    buffered_rowset_.clear();    
    // start working
    row_iter_->BeforeFirst();
    while (row_iter_->Next()) {
      const RowBatch &batch = row_iter_->Value();
      
    }
    row_iter_->BeforeFirst();
    size_t ktop = 0;
    while (row_iter_->Next()) {
      const RowBatch &batch = row_iter_->Value();
      for (size_t i = 0; i < batch.size; ++i) {
        if (ktop < buffered_rowset_.size() &&
            buffered_rowset_[ktop] == batch.base_rowid + i) {
          ++ktop;
          // TODO1
        }
      }
    }
    // sort columns
  }

 private:
  // row iterator
  utils::IIterator<RowBatch> *row_iter_;
  // column iterator
  ThreadColPageIterator *col_iter_;
  /*! \brief list of row index that are buffered */
  std::vector<bst_uint> buffered_rowset_;
};

}  // namespace io
}  // namespace xgboost
#endif  // XGBOOST_IO_PAGE_FMATRIX_INL_HPP_
