#ifndef XGBOOST_IO_PAGE_FMATRIX_INL_HPP_
#define XGBOOST_IO_PAGE_FMATRIX_INL_HPP_
/*!
 * \file page_fmatrix-inl.hpp
 *   col iterator based on sparse page
 * \author Tianqi Chen
 */
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
           page_->offset[i + 1] - page_->offset[i]);
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
    itr.get_factory().SetFile(fi, 0);
    itr.Init();
  }
  // set index set
  inline void SetIndexSet(const std::vector<bst_uint> &fset) {
    itr.get_factory().SetIndexSet(fset);
  }
  
 private:
  // output data
  ColBatch out_;
  SparsePage *page_;
  std::vector<SparseBatch::Inst> col_data_;
  utils::ThreadBuffer<SparsePage*, SparsePageFactory> itr;
};
/*!
 * \brief sparse matrix that support column access, CSC
 */
class FMatrixS : public IFMatrix {
 public:
  typedef SparseBatch::Entry Entry;
  /*! \brief constructor */
  FMatrixS(utils::IIterator<RowBatch> *iter) {
    this->iter_ = iter;
  }
  // destructor
  virtual ~FMatrixS(void) {
    if (iter_ != NULL) delete iter_;
  }
  /*! \return whether column access is enabled */
  virtual bool HaveColAccess(void) const {
    return col_ptr_.size() != 0;
  }
  /*! \brief get number of colmuns */
  virtual size_t NumCol(void) const {
    utils::Check(this->HaveColAccess(), "NumCol:need column access");
    return col_ptr_.size() - 1;
  }
  /*! \brief get number of buffered rows */
  virtual const std::vector<bst_uint> &buffered_rowset(void) const {
    return buffered_rowset_;
  }
  /*! \brief get column size */
  virtual size_t GetColSize(size_t cidx) const {
    return col_ptr_[cidx+1] - col_ptr_[cidx];
  }
  /*! \brief get column density */
  virtual float GetColDensity(size_t cidx) const {
    size_t nmiss = buffered_rowset_.size() - (col_ptr_[cidx+1] - col_ptr_[cidx]);
    return 1.0f - (static_cast<float>(nmiss)) / buffered_rowset_.size();
  }
  virtual void InitColAccess(const std::vector<bool> &enabled, 
                             float pkeep = 1.0f) {
    if (this->HaveColAccess()) return;
    this->InitColData(pkeep, enabled);
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
    col_iter_.SetBatch(col_ptr_, col_data_);
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
    col_iter_.SetBatch(col_ptr_, col_data_);
    return &col_iter_;
  }
};
}  // namespace io
}  // namespace xgboost
#endif  // XGBOOST_IO_PAGE_FMATRIX_INL_HPP_
