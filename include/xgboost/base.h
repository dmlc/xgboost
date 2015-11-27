/*!
 * Copyright (c) 2015 by Contributors
 * \file base.h
 * \brief defines configuration macros of xgboost.
 */
#ifndef XGBOOST_BASE_H_
#define XGBOOST_BASE_H_

#include <dmlc/base.h>

namespace xgboost {
/*!
 * \brief unsigned interger type used in boost,
 *  used for feature index and row index.
 */
typedef uint32_t bst_uint;
/*! \brief float type, used for storing statistics */
typedef float bst_float;
const float rt_eps = 1e-5f;
// min gap between feature values to allow a split happen
const float rt_2eps = rt_eps * 2.0f;

/*! \brief read-only sparse instance batch in CSR format */
struct SparseBatch {
  /*! \brief an entry of sparse vector */
  struct Entry {
    /*! \brief feature index */
    bst_uint index;
    /*! \brief feature value */
    bst_float fvalue;
    /*! \brief default constructor */
    Entry() {}
    /*!
     * \brief constructor with index and value
     * \param index The feature or row index.
     * \param fvalue THe feature value.
     */
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

}  // namespace xgboost
#endif  // XGBOOST_BASE_H_
