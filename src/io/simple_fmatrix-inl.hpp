#ifndef XGBOOST_IO_SIMPLE_FMATRIX_INL_HPP
#define XGBOOST_IO_SIMPLE_FMATRIX_INL_HPP
/*!
 * \file simple_fmatrix-inl.hpp
 * \brief the input data structure for gradient boosting
 * \author Tianqi Chen
 */
#include "../data.h"
#include "../utils/utils.h"
#include "../utils/random.h"
#include "../utils/omp.h"
#include "../utils/matrix_csr.h"
namespace xgboost {
namespace io {
/*!
 * \brief sparse matrix that support column access, CSC
 */
class FMatrixS : public IFMatrix{
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
  virtual void InitColAccess(float pkeep = 1.0f) {
    if (this->HaveColAccess()) return;
    this->InitColData(pkeep);
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
    col_iter_.col_index_ = fset;
    col_iter_.SetBatch(col_ptr_, col_data_);
    return &col_iter_;
  }
  /*!
   * \brief save column access data into stream
   * \param fo output stream to save to
   */
  inline void SaveColAccess(utils::IStream &fo) const {
    fo.Write(buffered_rowset_);
    if (buffered_rowset_.size() != 0) {
      SaveBinary(fo, col_ptr_, col_data_);
    }
  }
  /*!
   * \brief load column access data from stream
   * \param fo output stream to load from
   */
  inline void LoadColAccess(utils::IStream &fi) {
    utils::Check(fi.Read(&buffered_rowset_), "invalid input file format");
    if (buffered_rowset_.size() != 0) {
      LoadBinary(fi, &col_ptr_, &col_data_);
    }
  }
  /*!
   * \brief save data to binary stream
   * \param fo output stream
   * \param ptr pointer data
   * \param data data content
   */
  inline static void SaveBinary(utils::IStream &fo,
                                const std::vector<size_t> &ptr,
                                const std::vector<RowBatch::Entry> &data) {
    size_t nrow = ptr.size() - 1;
    fo.Write(&nrow, sizeof(size_t));
    fo.Write(BeginPtr(ptr), ptr.size() * sizeof(size_t));
    if (data.size() != 0) {
      fo.Write(BeginPtr(data), data.size() * sizeof(RowBatch::Entry));
    }
  }
  /*!
   * \brief load data from binary stream
   * \param fi input stream
   * \param out_ptr pointer data
   * \param out_data data content
   */
  inline static void LoadBinary(utils::IStream &fi,
                                std::vector<size_t> *out_ptr,
                                std::vector<RowBatch::Entry> *out_data) {
    size_t nrow;
    utils::Check(fi.Read(&nrow, sizeof(size_t)) != 0, "invalid input file format");
    out_ptr->resize(nrow + 1);
    utils::Check(fi.Read(BeginPtr(*out_ptr), out_ptr->size() * sizeof(size_t)) != 0,
                  "invalid input file format");
    out_data->resize(out_ptr->back());
    if (out_data->size() != 0) {
      utils::Assert(fi.Read(BeginPtr(*out_data), out_data->size() * sizeof(RowBatch::Entry)) != 0,
                    "invalid input file format");
    }
  }

 protected:
  /*!
   * \brief intialize column data
   * \param pkeep probability to keep a row
   */
  inline void InitColData(float pkeep) {
    buffered_rowset_.clear();
    // note: this part of code is serial, todo, parallelize this transformer
    utils::SparseCSRMBuilder<RowBatch::Entry> builder(col_ptr_, col_data_);
    builder.InitBudget(0);
    // start working
    iter_->BeforeFirst();
    while (iter_->Next()) {
      const RowBatch &batch = iter_->Value();
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
    builder.InitStorage();

    iter_->BeforeFirst();
    size_t ktop = 0;
    while (iter_->Next()) {
      const RowBatch &batch = iter_->Value();
      for (size_t i = 0; i < batch.size; ++i) {
        if (ktop < buffered_rowset_.size() &&
            buffered_rowset_[ktop] == batch.base_rowid+i) {
          ++ktop;
          RowBatch::Inst inst = batch[i];
          for (bst_uint j = 0; j < inst.length; ++j) {
            builder.PushElem(inst[j].index,
                             Entry((bst_uint)(batch.base_rowid+i),
                                   inst[j].fvalue));
          }
        }
      }
    }
    // sort columns
    bst_omp_uint ncol = static_cast<bst_omp_uint>(this->NumCol());
    #pragma omp parallel for schedule(static)
    for (bst_omp_uint i = 0; i < ncol; ++i) {
      std::sort(&col_data_[0] + col_ptr_[i],
                &col_data_[0] + col_ptr_[i + 1], Entry::CmpValue);
    }
  }

 private:
  // one batch iterator that return content in the matrix
  struct OneBatchIter: utils::IIterator<ColBatch> {
    OneBatchIter(void) : at_first_(true){}
    virtual ~OneBatchIter(void) {}
    virtual void BeforeFirst(void) {
      at_first_ = true;
    }
    virtual bool Next(void) {
      if (!at_first_) return false;
      at_first_ = false;
      return true;
    }
    virtual const ColBatch &Value(void) const {
      return batch_;
    }
    inline void SetBatch(const std::vector<size_t> &ptr,
                         const std::vector<ColBatch::Entry> &data) {
      batch_.size = col_index_.size();
      col_data_.resize(col_index_.size(), SparseBatch::Inst(NULL,0));
      for (size_t i = 0; i < col_data_.size(); ++i) {
        const bst_uint ridx = col_index_[i];
        col_data_[i] = SparseBatch::Inst(&data[0] + ptr[ridx],
                                         static_cast<bst_uint>(ptr[ridx+1] - ptr[ridx]));
      }
      batch_.col_index = BeginPtr(col_index_);
      batch_.col_data = BeginPtr(col_data_);
      this->BeforeFirst();
    }
    // data content
    std::vector<bst_uint> col_index_;
    std::vector<ColBatch::Inst> col_data_;
    // whether is at first
    bool at_first_;
    // temporal space for batch
    ColBatch batch_;
  }; 
  // --- data structure used to support InitColAccess --
  // column iterator
  OneBatchIter col_iter_;
  // row iterator
  utils::IIterator<RowBatch> *iter_;
  /*! \brief list of row index that are buffered */
  std::vector<bst_uint> buffered_rowset_;
  /*! \brief column pointer of CSC format */
  std::vector<size_t> col_ptr_;
  /*! \brief column datas in CSC format */
  std::vector<ColBatch::Entry> col_data_;
};
}  // namespace io
}  // namespace xgboost
#endif // XGBOOST_IO_SIMPLE_FMATRIX_INL_HPP
