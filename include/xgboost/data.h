/*!
 * Copyright (c) 2015 by Contributors
 * \file data.h
 * \brief The input data structure of xgboost.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_DATA_H_
#define XGBOOST_DATA_H_

#include <dmlc/base.h>
#include <dmlc/data.h>
#include <string>
#include <memory>
#include <vector>
#include "./base.h"

namespace xgboost {
// forward declare learner.
class LearnerImpl;

/*! \brief data type accepted by xgboost interface */
enum DataType {
  kFloat32 = 1,
  kDouble = 2,
  kUInt32 = 3,
  kUInt64 = 4
};

/*!
 * \brief Meta information about dataset, always sit in memory.
 */
struct MetaInfo {
  /*! \brief number of rows in the data */
  uint64_t num_row;
  /*! \brief number of columns in the data */
  uint64_t num_col;
  /*! \brief number of nonzero entries in the data */
  uint64_t num_nonzero;
  /*! \brief label of each instance */
  std::vector<bst_float> labels;
  /*!
   * \brief specified root index of each instance,
   *  can be used for multi task setting
   */
  std::vector<bst_uint> root_index;
  /*!
   * \brief the index of begin and end of a group
   *  needed when the learning task is ranking.
   */
  std::vector<bst_uint> group_ptr;
  /*! \brief weights of each instance, optional */
  std::vector<bst_float> weights;
  /*!
   * \brief initialized margins,
   * if specified, xgboost will start from this init margin
   * can be used to specify initial prediction to boost from.
   */
  std::vector<bst_float> base_margin;
  /*! \brief version flag, used to check version of this info */
  static const int kVersion = 1;
  /*! \brief default constructor */
  MetaInfo() : num_row(0), num_col(0), num_nonzero(0) {}
  /*!
   * \brief Get weight of each instances.
   * \param i Instance index.
   * \return The weight.
   */
  inline bst_float GetWeight(size_t i) const {
    return weights.size() != 0 ?  weights[i] : 1.0f;
  }
  /*!
   * \brief Get the root index of i-th instance.
   * \param i Instance index.
   * \return The pre-defined root index of i-th instance.
   */
  inline unsigned GetRoot(size_t i) const {
    return root_index.size() != 0 ? root_index[i] : 0U;
  }
  /*! \brief clear all the information */
  void Clear();
  /*!
   * \brief Load the Meta info from binary stream.
   * \param fi The input stream
   */
  void LoadBinary(dmlc::Stream* fi);
  /*!
   * \brief Save the Meta info to binary stream
   * \param fo The output stream.
   */
  void SaveBinary(dmlc::Stream* fo) const;
  /*!
   * \brief Set information in the meta info.
   * \param key The key of the information.
   * \param dptr The data pointer of the source array.
   * \param dtype The type of the source data.
   * \param num Number of elements in the source array.
   */
  void SetInfo(const char* key, const void* dptr, DataType dtype, size_t num);
};

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
    inline static bool CmpValue(const Entry& a, const Entry& b) {
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
    Inst() : data(0), length(0) {}
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
    return Inst(data_ptr + ind_ptr[i], static_cast<bst_uint>(ind_ptr[i + 1] - ind_ptr[i]));
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

/*!
 * \brief This is data structure that user can pass to DMatrix::Create
 *  to create a DMatrix for training, user can create this data structure
 *  for customized Data Loading on single machine.
 *
 *  On distributed setting, usually an customized dmlc::Parser is needed instead.
 */
class DataSource : public dmlc::DataIter<RowBatch> {
 public:
  /*!
   * \brief Meta information about the dataset
   * The subclass need to be able to load this correctly from data.
   */
  MetaInfo info;
};

/*!
 * \brief A vector-like structure to represent set of rows.
 * But saves the memory when all rows are in the set (common case in xgb)
 */
struct RowSet {
 public:
  /*! \return i-th row index */
  inline bst_uint operator[](size_t i) const;
  /*! \return the size of the set. */
  inline size_t size() const;
  /*! \brief push the index back to the set */
  inline void push_back(bst_uint i);
  /*! \brief clear the set */
  inline void clear();
  /*!
   * \brief save rowset to file.
   * \param fo The file to be saved.
   */
  inline void Save(dmlc::Stream* fo) const;
  /*!
   * \brief Load rowset from file.
   * \param fi The file to be loaded.
   * \return if read is successful.
   */
  inline bool Load(dmlc::Stream* fi);
  /*! \brief constructor */
  RowSet() : size_(0) {}

 private:
  /*! \brief The internal data structure of size */
  uint64_t size_;
  /*! \brief The internal data structure of row set if not all*/
  std::vector<bst_uint> rows_;
};

/*!
 * \brief Internal data structured used by XGBoost during training.
 *  There are two ways to create a customized DMatrix that reads in user defined-format.
 *
 *  - Provide a dmlc::Parser and pass into the DMatrix::Create
 *  - Alternatively, if data can be represented by an URL, define a new dmlc::Parser and register by DMLC_REGISTER_DATA_PARSER;
 *      - This works best for user defined data input source, such as data-base, filesystem.
 *  - Provide a DataSource, that can be passed to DMatrix::Create
 *      This can be used to re-use inmemory data structure into DMatrix.
 */
class DMatrix {
 public:
  /*! \brief default constructor */
  DMatrix() : cache_learner_ptr_(nullptr) {}
  /*! \brief meta information of the dataset */
  virtual MetaInfo& info() = 0;
  /*! \brief meta information of the dataset */
  virtual const MetaInfo& info() const = 0;
  /*!
   * \brief get the row iterator, reset to beginning position
   * \note Only either RowIterator or  column Iterator can be active.
   */
  virtual dmlc::DataIter<RowBatch>* RowIterator() = 0;
  /*!\brief get column iterator, reset to the beginning position */
  virtual dmlc::DataIter<ColBatch>* ColIterator() = 0;
  /*!
   * \brief get the column iterator associated with subset of column features.
   * \param fset is the list of column index set that must be contained in the returning Column iterator
   * \return the column iterator, initialized so that it reads the elements in fset
   */
  virtual dmlc::DataIter<ColBatch>* ColIterator(const std::vector<bst_uint>& fset) = 0;
  /*!
   * \brief check if column access is supported, if not, initialize column access.
   * \param enabled whether certain feature should be included in column access.
   * \param subsample subsample ratio when generating column access.
   * \param max_row_perbatch auxiliary information, maximum row used in each column batch.
   *         this is a hint information that can be ignored by the implementation.
   * \return Number of column blocks in the column access.
   */
  virtual void InitColAccess(const std::vector<bool>& enabled,
                             float subsample,
                             size_t max_row_perbatch) = 0;
  // the following are column meta data, should be able to answer them fast.
  /*! \return whether column access is enabled */
  virtual bool HaveColAccess() const = 0;
  /*! \return Whether the data columns single column block. */
  virtual bool SingleColBlock() const = 0;
  /*! \brief get number of non-missing entries in column */
  virtual size_t GetColSize(size_t cidx) const = 0;
  /*! \brief get column density */
  virtual float GetColDensity(size_t cidx) const = 0;
  /*! \return reference of buffered rowset, in column access */
  virtual const RowSet& buffered_rowset() const = 0;
  /*! \brief virtual destructor */
  virtual ~DMatrix() {}
  /*!
   * \brief Save DMatrix to local file.
   *  The saved file only works for non-sharded dataset(single machine training).
   *  This API is deprecated and dis-encouraged to use.
   * \param fname The file name to be saved.
   * \return The created DMatrix.
   */
  virtual void SaveToLocalFile(const std::string& fname);
  /*!
   * \brief Load DMatrix from URI.
   * \param uri The URI of input.
   * \param silent Whether print information during loading.
   * \param load_row_split Flag to read in part of rows, divided among the workers in distributed mode.
   * \param file_format The format type of the file, used for dmlc::Parser::Create.
   *   By default "auto" will be able to load in both local binary file.
   * \return The created DMatrix.
   */
  static DMatrix* Load(const std::string& uri,
                       bool silent,
                       bool load_row_split,
                       const std::string& file_format = "auto");
  /*!
   * \brief create a new DMatrix, by wrapping a row_iterator, and meta info.
   * \param source The source iterator of the data, the create function takes ownership of the source.
   * \param cache_prefix The path to prefix of temporary cache file of the DMatrix when used in external memory mode.
   *     This can be nullptr for common cases, and in-memory mode will be used.
   * \return a Created DMatrix.
   */
  static DMatrix* Create(std::unique_ptr<DataSource>&& source,
                         const std::string& cache_prefix = "");
  /*!
   * \brief Create a DMatrix by loading data from parser.
   *  Parser can later be deleted after the DMatrix i created.
   * \param parser The input data parser
   * \param cache_prefix The path to prefix of temporary cache file of the DMatrix when used in external memory mode.
   *     This can be nullptr for common cases, and in-memory mode will be used.
   * \sa dmlc::Parser
   * \note dmlc-core provides efficient distributed data parser for libsvm format.
   *  User can create and register customized parser to load their own format using DMLC_REGISTER_DATA_PARSER.
   *  See "dmlc-core/include/dmlc/data.h" for detail.
   * \return A created DMatrix.
   */
  static DMatrix* Create(dmlc::Parser<uint32_t>* parser,
                         const std::string& cache_prefix = "");

 private:
  // allow learner class to access this field.
  friend class LearnerImpl;
  /*! \brief public field to back ref cached matrix. */
  LearnerImpl* cache_learner_ptr_;
};

// implementation of inline functions
inline bst_uint RowSet::operator[](size_t i) const {
  return rows_.size() == 0 ? i : rows_[i];
}

inline size_t RowSet::size() const {
  return size_;
}

inline void RowSet::clear() {
  rows_.clear(); size_ = 0;
}

inline void RowSet::push_back(bst_uint i) {
  if (rows_.size() == 0) {
    if (i == size_) {
      ++size_; return;
    } else {
      rows_.resize(size_);
      for (size_t i = 0; i < size_; ++i) {
        rows_[i] = static_cast<bst_uint>(i);
      }
    }
  }
  rows_.push_back(i);
  ++size_;
}

inline void RowSet::Save(dmlc::Stream* fo) const {
  fo->Write(rows_);
  fo->Write(&size_, sizeof(size_));
}

inline bool RowSet::Load(dmlc::Stream* fi) {
  if (!fi->Read(&rows_)) return false;
  if (rows_.size() != 0) return true;
  return fi->Read(&size_, sizeof(size_)) == sizeof(size_);
}
}  // namespace xgboost

namespace dmlc {
DMLC_DECLARE_TRAITS(is_pod, xgboost::SparseBatch::Entry, true);
DMLC_DECLARE_TRAITS(has_saveload, xgboost::RowSet, true);
}
#endif  // XGBOOST_DATA_H_
