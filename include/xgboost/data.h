/*!
 * Copyright (c) 2014 by Contributors
 * \file data.h
 * \brief The input data structure for gradient boosting.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_DATA_H_
#define XGBOOST_DATA_H_

#include <dmlc/base.h>
#include <dmlc/data.h>
#include <memory>
#include "./base.h"

namespace xgboost {
// forward declare learner.
class Learner;

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
  size_t num_row;
  /*! \brief number of columns in the data */
  size_t num_col;
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
  static const int kVersion = 0;
  /*! \brief default constructor */
  MetaInfo() : num_row(0), num_col(0) {}
  /*!
   * \brief Get weight of each instances.
   * \param i Instance index.
   * \return The weight.
   */
  inline float GetWeight(size_t i) const {
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
  void LoadBinary(dmlc::Stream *fi);
  /*!
   * \brief Save the Meta info to binary stream
   * \param fo The output stream.
   */
  void SaveBinary(dmlc::Stream *fo) const;
  /*!
   * \brief Set information in the meta info.
   * \param key The key of the information.
   * \param dptr The data pointer of the source array.
   * \param dtype The type of the source data.
   * \param num Number of elements in the source array.
   */
  void SetInfo(const char* key, const void* dptr, DataType dtype, size_t num);
  /*!
   * \brief Get information from meta info.
   * \param key The key of the information.
   * \param dptr The output data pointer of the source array.
   * \param dtype The output data type of the information array.
   * \param num Number of elements in the array.
   */
  void GetInfo(const char* key, const void** dptr, DataType* dtype, size_t* num) const;
};

/*!
 * \brief This is data structure that user can pass to DMatrix::Create
 *  to create a DMatrix for training, user can create this data structure
 *  for customized Data Loading on single machine.
 */
struct DataSource {
  /*!
   * \brief Used to initialize the meta information of DMatrix
   *  The created DMatrix can change its own info later.
   */
  MetaInfo info;
  /*!
   * \brief Used for row based iteration of DMatrix,
   */
  std::unique_ptr<dmlc::DataIter<RowBatch> > row_iter;
};

/*!
 * \brief Internal data structured used by XGBoost during training.
 *  There are two ways to create a customized DMatrix that reads in user defined-format.
 *
 *  - Define a new dmlc::Parser and register by DMLC_REGISTER_DATA_PARSER;
 *      This works best for user defined data input source, such as data-base, filesystem.
 *  - Provdie a DataSource, that can be passed to DMatrix::Create
 *      This can be used to re-use inmemory data structure into DMatrix.
 */
class DMatrix {
 public:
  /*! \brief meta information that is always stored in DMatrix */
  MetaInfo info;
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
   * \param max_row_perbatch auxilary information, maximum row used in each column batch.
   *         this is a hint information that can be ignored by the implementation.
   */
  virtual void InitColAccess(const std::vector<bool>& enabled,
                             float subsample,
                             size_t max_row_perbatch) = 0;
  // the following are column meta data, should be able to answer them fast.
  /*! \return whether column access is enabled */
  virtual bool HaveColAccess() const = 0;
  /*! \brief get number of non-missing entries in column */
  virtual size_t GetColSize(size_t cidx) const = 0;
  /*! \brief get column density */
  virtual float GetColDensity(size_t cidx) const = 0;
  /*! \return reference of buffered rowset, in column access */
  virtual const std::vector<bst_uint> &buffered_rowset() const = 0;
  /*! \brief virtual destructor */
  virtual ~DMatrix() {}
  /*!
   * \brief Save DMatrix to local file.
   *  The saved file only works for non-sharded dataset(single machine training).
   * \param fname The file name to be saved.
   * \return The created DMatrix.
   */
  virtual void SaveToLocalFile(const char* fname);
  /*!
   * \brief Load DMatrix from URI.
   * \param uri The URI of input.
   * \param silent Whether print information during loading.
   * \param load_row_split Flag to read in part of rows, divided among the workers in distributed mode.
   * \return The created DMatrix.
   */
  static DMatrix* Load(const char* uri,
                       bool silent,
                       bool load_row_split);
  /*!
   * \brief create a new DMatrix, by wrapping a row_iterator, and meta info.
   * \param source The source iterator of the data, the create function takes ownership of the source.
   * \param info The meta information in the DMatrix, need to move ownership to DMatrix.
   * \param cache_prefix The path to prefix of temporary cache file of the DMatrix when used in external memory mode.
   *     This can be nullptr for common cases, and in-memory mode will be used.
   * \return a Created DMatrix.
   */
  static DMatrix* Create(DataSource&& source,
                         const char* cache_prefix=nullptr);
  /*!
   * \brief Create a DMatrix by loaidng data from parser.
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
                         const char* cache_prefix=nullptr);
};
}  // namespace xgboost
#endif  // XGBOOST_DATA_H_
