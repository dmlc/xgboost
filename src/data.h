/*!
 * Copyright (c) 2014 by Contributors
 * \file data.h
 * \brief the input data structure for gradient boosting
 * \author Tianqi Chen
 */
#ifndef XGBOOST_DATA_H_
#define XGBOOST_DATA_H_

#include <cstdio>
#include <vector>
#include "utils/utils.h"
#include "utils/iterator.h"

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
struct bst_gpair {
  /*! \brief gradient statistics */
  bst_float grad;
  /*! \brief second order gradient statistics */
  bst_float hess;
  bst_gpair(void) {}
  bst_gpair(bst_float grad, bst_float hess) : grad(grad), hess(hess) {}
};

/*!
 * \brief extra information that might needed by gbm and tree module
 * these information are not necessarily presented, and can be empty
 */
struct BoosterInfo {
  /*! \brief number of rows in the data */
  size_t num_row;
  /*! \brief number of columns in the data */
  size_t num_col;
  /*!
   * \brief specified root index of each instance,
   *  can be used for multi task setting
   */
  std::vector<unsigned> root_index;
  /*! \brief set fold indicator */
  std::vector<unsigned> fold_index;
  /*! \brief number of rows, number of columns */
  BoosterInfo(void) : num_row(0), num_col(0) {
  }
  /*! \brief get root of ith instance */
  inline unsigned GetRoot(size_t i) const {
    return root_index.size() == 0 ? 0 : root_index[i];
  }
};

/*! \brief read-only sparse instance batch in CSR format */
struct SparseBatch {
  /*! \brief an entry of sparse vector */
  struct Entry {
    /*! \brief feature index */
    bst_uint index;
    /*! \brief feature value */
    bst_float fvalue;
    // default constructor
    Entry(void) {}
    Entry(bst_uint index, bst_float fvalue) : index(index), fvalue(fvalue) {}
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
    bst_uint length;
    /*! \brief constructor */
    Inst(const Entry *data, bst_uint length) : data(data), length(length) {}
    /*! \brief get i-th pair in the sparse vector*/
    inline const Entry& operator[](size_t i) const {
      return data[i];
    }
  };
  /*! \brief batch size */
  size_t size;
};
/*! \brief read-only row batch, used to access row continuously */
struct RowBatch : public SparseBatch {
  /*! \brief the offset of rowid of this batch */
  size_t base_rowid;
  /*! \brief array[size+1], row pointer of each of the elements */
  const size_t *ind_ptr;
  /*! \brief array[ind_ptr.back()], content of the sparse element */
  const Entry *data_ptr;
  /*! \brief get i-th row from the batch */
  inline Inst operator[](size_t i) const {
    return Inst(data_ptr + ind_ptr[i], static_cast<bst_uint>(ind_ptr[i+1] - ind_ptr[i]));
  }
};
/*!
 * \brief read-only column batch, used to access columns,
 * the columns are not required to be continuous
 */
struct ColBatch : public SparseBatch {
  /*! \brief column index of each columns in the data */
  const bst_uint *col_index;
  /*! \brief pointer to the column data */
  const Inst *col_data;
  /*! \brief get i-th column from the batch */
  inline Inst operator[](size_t i) const {
    return col_data[i];
  }
};
/**
 * \brief interface of feature matrix, needed for tree construction
 *  this interface defines two way to access features,
 *  row access is defined by iterator of RowBatch
 *  col access is optional, checked by HaveColAccess, and defined by iterator of ColBatch
 */
class IFMatrix {
 public:
  // the interface only need to ganrantee row iter
  // column iter is active, when ColIterator is called, row_iter can be disabled
  /*! \brief get the row iterator associated with FMatrix */
  virtual utils::IIterator<RowBatch> *RowIterator(void) = 0;
  /*!\brief get column iterator */
  virtual utils::IIterator<ColBatch> *ColIterator(void) = 0;
  /*!
   * \brief get the column iterator associated with FMatrix with subset of column features
   * \param fset is the list of column index set that must be contained in the returning Column iterator
   * \return the column iterator, initialized so that it reads the elements in fset
   */
  virtual utils::IIterator<ColBatch> *ColIterator(const std::vector<bst_uint> &fset) = 0;
  /*!
   * \brief check if column access is supported, if not, initialize column access
   * \param enabled whether certain feature should be included in column access
   * \param subsample subsample ratio when generating column access
   * \param max_row_perbatch auxilary information, maximum row used in each column batch
   *         this is a hint information that can be ignored by the implementation
   */
  virtual void InitColAccess(const std::vector<bool> &enabled,
                             float subsample,
                             size_t max_row_perbatch) = 0;
  // the following are column meta data, should be able to answer them fast
  /*! \return whether column access is enabled */
  virtual bool HaveColAccess(void) const = 0;
  /*! \return number of columns in the FMatrix */
  virtual size_t NumCol(void) const = 0;
  /*! \brief get number of non-missing entries in column */
  virtual size_t GetColSize(size_t cidx) const = 0;
  /*! \brief get column density */
  virtual float GetColDensity(size_t cidx) const = 0;
  /*! \brief reference of buffered rowset */
  virtual const std::vector<bst_uint> &buffered_rowset(void) const = 0;
  // virtual destructor
  virtual ~IFMatrix(void){}
};
}  // namespace xgboost
#endif  // XGBOOST_DATA_H_
