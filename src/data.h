#ifndef XGBOOST_UNITY_DATA_H
#define XGBOOST_UNITY_DATA_H
/*!
 * \file data.h
 * \brief the input data structure for gradient boosting
 * \author Tianqi Chen
 */
#include <cstdio>
#include <vector>
#include <limits>
#include <algorithm>
#include "utils/io.h"
#include "utils/utils.h"
#include "utils/iterator.h"
#include "utils/matrix_csr.h"

namespace xgboost {
/*! 
 * \brief unsigned interger type used in boost, 
 *        used for feature index and row index 
 */
typedef unsigned bst_uint;
/*! \brief float type, used for storing statistics */
typedef float bst_float;
const float rt_eps = 1e-5f;
// min gap between feature values to allow a split happen
const float rt_2eps = rt_eps * 2.0f;

/*! \brief gradient statistics pair usually needed in gradient boosting */
struct bst_gpair{
  /*! \brief gradient statistics */
  bst_float grad;
  /*! \brief second order gradient statistics */
  bst_float hess;
  bst_gpair(void) {}
  bst_gpair(bst_float grad, bst_float hess) : grad(grad), hess(hess) {}
};

/*! \brief read-only sparse instance batch in CSR format */
struct SparseBatch {
  /*! \brief an entry of sparse vector */
  struct Entry {
    /*! \brief feature index */
    bst_uint findex;
    /*! \brief feature value */
    bst_float fvalue;
    // default constructor
    Entry(void) {}
    Entry(bst_uint findex, bst_float fvalue) : findex(findex), fvalue(fvalue) {}
    /*! \brief reversely compare feature values */
    inline static bool CmpValue(const Entry &a, const Entry &b) {
      return a.fvalue < b.fvalue;
    }
  };
  /*! \brief an instance of sparse vector in the batch */
  struct Inst {
    /*! \brief pointer to the elements*/
    const Entry *data;
    /*! \brief length of the instance */
    const bst_uint length;
    /*! \brief constructor */
    Inst(const Entry *data, bst_uint length) : data(data), length(length) {}
    /*! \brief get i-th pair in the sparse vector*/
    inline const Entry& operator[](size_t i) const {
      return data[i];
    }
  };
  /*! \brief batch size */
  size_t size;
  /*! \brief the offset of rowid of this batch */
  size_t base_rowid;
  /*! \brief array[size+1], row pointer of each of the elements */
  const size_t *row_ptr;
  /*! \brief array[row_ptr.back()], content of the sparse element */
  const Entry *data_ptr;
  /*! \brief get i-th row from the batch */
  inline Inst operator[](size_t i) const {
    return Inst(data_ptr + row_ptr[i], row_ptr[i+1] - row_ptr[i]);
  }
};

/**
 * \brief This is a interface convention via template, defining the way to access features,
 *        column access rule is defined by template, for efficiency purpose, 
 *        row access is defined by iterator of sparse batches
 * \tparam Derived type of actual implementation
 */
template<typename Derived>
class FMatrixInterface {
 public:
  /*! \brief example iterator over one column */
  struct ColIter{
    /*!
     * \brief move to next position
     * \return whether there is element in next position
     */
    inline bool Next(void);
    /*! \return row index of current position  */
    inline bst_uint rindex(void) const;
    /*! \return feature value in current position */
    inline bst_float fvalue(void) const;
  };
  /*! \brief backward iterator over column */
  struct ColBackIter : public ColIter {};
 public:
  // column access is needed by some of tree construction algorithms
  /*!
   * \brief get column iterator, the columns must be sorted by feature value
   * \param cidx column index
   * \return column iterator
   */
  inline ColIter GetSortedCol(size_t cidx) const;
  /*!
   * \brief get column backward iterator, starts from biggest fvalue, and iterator back
   * \param cidx column index
   * \return reverse column iterator
   */
  inline ColBackIter GetReverseSortedCol(size_t cidx) const;
  /*!
   * \brief get number of columns
   * \return number of columns
   */
  inline size_t NumCol(void) const;
  /*! 
   * \brief check if column access is supported, if not, initialize column access 
   * \param max_rows maximum number of rows allowed in constructor 
   */
  inline void InitColAccess(void);
  /*! \return whether column access is enabled */
  inline bool HaveColAccess(void) const;
  /*! \breif return #entries-in-col */
  inline size_t GetColSize(size_t cidx) const;
  /*!
   * \breif return #entries-in-col / #rows
   * \param cidx column index 
   *   this function is used to help speedup, 
   *   doese not necessarily implement it if not sure, return 0.0;
   * \return column density
   */
  inline float GetColDensity(size_t cidx) const;
  /*! \brief get the row iterator associated with FMatrix */
  virtual utils::IIterator<SparseBatch>* RowIterator(void) const = 0;
};

/*!
 * \brief sparse matrix that support column access, CSC
 */
class FMatrixS : public FMatrixInterface<FMatrixS>{
 public:
  typedef SparseBatch::Entry Entry;
  /*! \brief row iterator */
  struct ColIter{
    const Entry *dptr_, *end_;
    ColIter(const Entry* begin, const Entry* end)
        :dptr_(begin), end_(end) {}
    inline bool Next(void) {
      if (dptr_ == end_) {
        return false;
      } else {
        ++dptr_; return true;
      }
    }
    inline bst_uint rindex(void) const {
      return dptr_->findex;
    }
    inline bst_float fvalue(void) const {
      return dptr_->fvalue;
    }
  };
  /*! \brief reverse column iterator */
  struct ColBackIter : public ColIter {
    ColBackIter(const Entry* dptr, const Entry* end) : ColIter(dptr, end) {}
    // shadows ColIter::Next
    inline bool Next(void) {
      if (dptr_ == end_) {
        return false;
      } else {
        --dptr_; return true;
      }
    }
  };
  /*! \brief constructor */
  explicit FMatrixS(utils::IIterator<SparseBatch> *base_iter)
      : iter_(base_iter) {}
  // destructor
  virtual ~FMatrixS(void) {
    delete iter_;
  }
  /*! \return whether column access is enabled */
  inline bool HaveColAccess(void) const {
    return col_ptr_.size() != 0;
  }
  /*! \brief get number of colmuns */
  inline size_t NumCol(void) const {
    utils::Check(this->HaveColAccess(), "NumCol:need column access");
    return col_ptr_.size() - 1;
  }
  /*! \brief get col sorted iterator */
  inline ColIter GetSortedCol(size_t cidx) const {
    utils::Assert(cidx < this->NumCol(), "col id exceed bound");
    return ColIter(&col_data_[col_ptr_[cidx]] - 1,
                   &col_data_[col_ptr_[cidx + 1]] - 1);
  }
  /*! 
   * \brief get reversed col iterator, 
   *   this function will be deprecated at some point 
   */
  inline ColBackIter GetReverseSortedCol(size_t cidx) const {
    utils::Assert(cidx < this->NumCol(), "col id exceed bound");
    return ColBackIter(&col_data_[col_ptr_[cidx + 1]],
                       &col_data_[col_ptr_[cidx]]);
  }
  /*! \brief get col size */
  inline size_t GetColSize(size_t cidx) const {
    return col_ptr_[cidx+1] - col_ptr_[cidx];
  }
  /*! \brief get column density */
  inline float GetColDensity(size_t cidx) const {
    size_t nmiss = num_buffered_row_ - (col_ptr_[cidx+1] - col_ptr_[cidx]);
    return 1.0f - (static_cast<float>(nmiss)) / num_buffered_row_;
  }
  virtual void InitColAccess(void) {
    if (this->HaveColAccess()) return;
    const size_t max_nrow = std::numeric_limits<bst_uint>::max();
    this->InitColData(max_nrow);
  }
  /*! \brief get the row iterator associated with FMatrix */
  virtual utils::IIterator<SparseBatch>* RowIterator(void) const {
    return iter_;
  }

 protected:
  /*!
   * \brief intialize column data 
   * \param max_nrow maximum number of rows supported 
   */
  inline void InitColData(size_t max_nrow) {
    // note: this part of code is serial, todo, parallelize this transformer
    utils::SparseCSRMBuilder<SparseBatch::Entry> builder(col_ptr_, col_data_);
    builder.InitBudget(0);
    // start working
    iter_->BeforeFirst();
    num_buffered_row_ = 0;
    while (iter_->Next()) {
      const SparseBatch &batch = iter_->Value();
      if (batch.base_rowid >= max_nrow) break;
      const size_t nbatch = std::min(batch.size, max_nrow - batch.base_rowid);
      for (size_t i = 0; i < nbatch; ++i, ++num_buffered_row_) {
        SparseBatch::Inst inst = batch[i];
        for (bst_uint j = 0; j < batch.size; ++j) {
          builder.AddBudget(inst[j].findex);
        }
      }
    }

    builder.InitStorage();

    iter_->BeforeFirst();
    while (iter_->Next()) {
      const SparseBatch &batch = iter_->Value();
      if (batch.base_rowid >= max_nrow) break;
      const size_t nbatch = std::min(batch.size, max_nrow - batch.base_rowid);
      for (size_t i = 0; i < nbatch; ++i) {
        SparseBatch::Inst inst = batch[i];
        for (bst_uint j = 0; j < batch.size; ++j) {
          builder.PushElem(inst[j].findex,
                           Entry((bst_uint)(batch.base_rowid+j),
                                 inst[j].fvalue));
        }
      }
    }

    // sort columns
    unsigned ncol = static_cast<unsigned>(this->NumCol());
    #pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < ncol; ++i) {
      std::sort(&col_data_[col_ptr_[i]],
                &col_data_[col_ptr_[i + 1]], Entry::CmpValue);
    }
  }

 private:
  // --- data structure used to support InitColAccess --
  utils::IIterator<SparseBatch> *iter_;
  /*! \brief number */
  size_t num_buffered_row_;
  /*! \brief column pointer of CSC format */
  std::vector<size_t>  col_ptr_;
  /*! \brief column datas in CSC format */
  std::vector<SparseBatch::Entry>  col_data_;
};
}  // namespace xgboost
#endif
