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
#include <cstring>
#include <memory>
#include <numeric>
#include <algorithm>
#include <string>
#include <vector>
#include "./base.h"
#include "../../src/common/span.h"
#include "../../src/common/group_data.h"

#include "../../src/common/host_device_vector.h"

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
class MetaInfo {
 public:
  /*! \brief number of rows in the data */
  uint64_t num_row_{0};
  /*! \brief number of columns in the data */
  uint64_t num_col_{0};
  /*! \brief number of nonzero entries in the data */
  uint64_t num_nonzero_{0};
  /*! \brief label of each instance */
  HostDeviceVector<bst_float> labels_;
  /*!
   * \brief specified root index of each instance,
   *  can be used for multi task setting
   */
  std::vector<bst_uint> root_index_;
  /*!
   * \brief the index of begin and end of a group
   *  needed when the learning task is ranking.
   */
  std::vector<bst_uint> group_ptr_;
  /*! \brief weights of each instance, optional */
  HostDeviceVector<bst_float> weights_;
  /*! \brief session-id of each instance, optional */
  std::vector<uint64_t> qids_;
  /*!
   * \brief initialized margins,
   * if specified, xgboost will start from this init margin
   * can be used to specify initial prediction to boost from.
   */
  HostDeviceVector<bst_float> base_margin_;
  /*! \brief version flag, used to check version of this info */
  static const int kVersion = 2;
  /*! \brief version that introduced qid field */
  static const int kVersionQidAdded = 2;
  /*! \brief default constructor */
  MetaInfo()  = default;
  /*!
   * \brief Get weight of each instances.
   * \param i Instance index.
   * \return The weight.
   */
  inline bst_float GetWeight(size_t i) const {
    return weights_.Size() != 0 ?  weights_.HostVector()[i] : 1.0f;
  }
  /*!
   * \brief Get the root index of i-th instance.
   * \param i Instance index.
   * \return The pre-defined root index of i-th instance.
   */
  inline unsigned GetRoot(size_t i) const {
    return root_index_.size() != 0 ? root_index_[i] : 0U;
  }
  /*! \brief get sorted indexes (argsort) of labels by absolute value (used by cox loss) */
  inline const std::vector<size_t>& LabelAbsSort() const {
    if (label_order_cache_.size() == labels_.Size()) {
      return label_order_cache_;
    }
    label_order_cache_.resize(labels_.Size());
    std::iota(label_order_cache_.begin(), label_order_cache_.end(), 0);
    const auto& l = labels_.HostVector();
    XGBOOST_PARALLEL_SORT(label_order_cache_.begin(), label_order_cache_.end(),
              [&l](size_t i1, size_t i2) {return std::abs(l[i1]) < std::abs(l[i2]);});

    return label_order_cache_;
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

 private:
  /*! \brief argsort of labels */
  mutable std::vector<size_t> label_order_cache_;
};

/*! \brief Element from a sparse vector */
struct Entry {
  /*! \brief feature index */
  bst_uint index;
  /*! \brief feature value */
  bst_float fvalue;
  /*! \brief default constructor */
  Entry() = default;
  /*!
   * \brief constructor with index and value
   * \param index The feature or row index.
   * \param fvalue The feature value.
   */
  Entry(bst_uint index, bst_float fvalue) : index(index), fvalue(fvalue) {}
  /*! \brief reversely compare feature values */
  inline static bool CmpValue(const Entry& a, const Entry& b) {
    return a.fvalue < b.fvalue;
  }
  inline bool operator==(const Entry& other) const {
    return (this->index == other.index && this->fvalue == other.fvalue);
  }
};

/*!
 * \brief In-memory storage unit of sparse batch, stored in CSR format.
 */
class SparsePage {
 public:
  // Offset for each row.
  HostDeviceVector<size_t> offset;
  /*! \brief the data of the segments */
  HostDeviceVector<Entry> data;

  size_t base_rowid;

  /*! \brief an instance of sparse vector in the batch */
  using Inst = common::Span<Entry const>;

  /*! \brief get i-th row from the batch */
  inline Inst operator[](size_t i) const {
    const auto& data_vec = data.HostVector();
    const auto& offset_vec = offset.HostVector();
    return {data_vec.data() + offset_vec[i],
            static_cast<Inst::index_type>(offset_vec[i + 1] - offset_vec[i])};
  }

  /*! \brief constructor */
  SparsePage() {
    this->Clear();
  }
  /*! \return number of instance in the page */
  inline size_t Size() const {
    return offset.Size() - 1;
  }
  /*! \return estimation of memory cost of this page */
  inline size_t MemCostBytes() const {
    return offset.Size() * sizeof(size_t) + data.Size() * sizeof(Entry);
  }
  /*! \brief clear the page */
  inline void Clear() {
    base_rowid = 0;
    auto& offset_vec = offset.HostVector();
    offset_vec.clear();
    offset_vec.push_back(0);
    data.HostVector().clear();
  }

  SparsePage GetTranspose(int num_columns) const {
    SparsePage transpose;
    common::ParallelGroupBuilder<Entry> builder(&transpose.offset.HostVector(),
                                                &transpose.data.HostVector());
    const int nthread = omp_get_max_threads();
    builder.InitBudget(num_columns, nthread);
    long batch_size = static_cast<long>(this->Size());  // NOLINT(*)
#pragma omp parallel for schedule(static)
    for (long i = 0; i < batch_size; ++i) {  // NOLINT(*)
      int tid = omp_get_thread_num();
      auto inst = (*this)[i];
      for (bst_uint j = 0; j < inst.size(); ++j) {
        builder.AddBudget(inst[j].index, tid);
      }
    }
    builder.InitStorage();
#pragma omp parallel for schedule(static)
    for (long i = 0; i < batch_size; ++i) {  // NOLINT(*)
      int tid = omp_get_thread_num();
      auto inst = (*this)[i];
      for (bst_uint j = 0; j < inst.size(); ++j) {
        builder.Push(
            inst[j].index,
            Entry(static_cast<bst_uint>(this->base_rowid + i), inst[j].fvalue),
            tid);
      }
    }
    return transpose;
  }

  void SortRows() {
    auto ncol = static_cast<bst_omp_uint>(this->Size());
#pragma omp parallel for schedule(dynamic, 1)
    for (bst_omp_uint i = 0; i < ncol; ++i) {
      if (this->offset.HostVector()[i] < this->offset.HostVector()[i + 1]) {
        std::sort(
            this->data.HostVector().begin() + this->offset.HostVector()[i],
            this->data.HostVector().begin() + this->offset.HostVector()[i + 1],
            Entry::CmpValue);
      }
    }
  }

  /*!
   * \brief Push row block into the page.
   * \param batch the row batch.
   */
  inline void Push(const dmlc::RowBlock<uint32_t>& batch) {
    auto& data_vec = data.HostVector();
    auto& offset_vec = offset.HostVector();
    data_vec.reserve(data.Size() + batch.offset[batch.size] - batch.offset[0]);
    offset_vec.reserve(offset.Size() + batch.size);
    CHECK(batch.index != nullptr);
    for (size_t i = 0; i < batch.size; ++i) {
      offset_vec.push_back(offset_vec.back() + batch.offset[i + 1] - batch.offset[i]);
    }
    for (size_t i = batch.offset[0]; i < batch.offset[batch.size]; ++i) {
      uint32_t index = batch.index[i];
      bst_float fvalue = batch.value == nullptr ? 1.0f : batch.value[i];
      data_vec.emplace_back(index, fvalue);
    }
    CHECK_EQ(offset_vec.back(), data.Size());
  }
  /*!
   * \brief Push a sparse page
   * \param batch the row page
   */
  inline void Push(const SparsePage &batch) {
    auto& data_vec = data.HostVector();
    auto& offset_vec = offset.HostVector();
    const auto& batch_offset_vec = batch.offset.HostVector();
    const auto& batch_data_vec = batch.data.HostVector();
    size_t top = offset_vec.back();
    data_vec.resize(top + batch.data.Size());
    std::memcpy(dmlc::BeginPtr(data_vec) + top,
                dmlc::BeginPtr(batch_data_vec),
                sizeof(Entry) * batch.data.Size());
    size_t begin = offset.Size();
    offset_vec.resize(begin + batch.Size());
    for (size_t i = 0; i < batch.Size(); ++i) {
      offset_vec[i + begin] = top + batch_offset_vec[i + 1];
    }
  }
  /*!
   * \brief Push one instance into page
   *  \param inst an instance row
   */
  inline void Push(const Inst &inst) {
    auto& data_vec = data.HostVector();
    auto& offset_vec = offset.HostVector();
    offset_vec.push_back(offset_vec.back() + inst.size());

    size_t begin = data_vec.size();
    data_vec.resize(begin + inst.size());
    if (inst.size() != 0) {
      std::memcpy(dmlc::BeginPtr(data_vec) + begin, inst.data(),
                  sizeof(Entry) * inst.size());
    }
  }

  size_t Size() { return offset.Size() - 1; }
};

class BatchIteratorImpl {
 public:
  virtual ~BatchIteratorImpl() {}
  virtual BatchIteratorImpl* Clone() = 0;
  virtual const SparsePage& operator*() const = 0;
  virtual void operator++() = 0;
  virtual bool AtEnd() const = 0;
};

class BatchIterator {
 public:
  using iterator_category = std::forward_iterator_tag;
  explicit BatchIterator(BatchIteratorImpl* impl) { impl_.reset(impl); }

  BatchIterator(const BatchIterator& other) {
    if (other.impl_) {
      impl_.reset(other.impl_->Clone());
    } else {
      impl_.reset();
    }
  }

  void operator++() {
    CHECK(impl_ != nullptr);
    ++(*impl_);
  }

  const SparsePage& operator*() const {
    CHECK(impl_ != nullptr);
    return *(*impl_);
  }

  bool operator!=(const BatchIterator& rhs) const {
    CHECK(impl_ != nullptr);
    return !impl_->AtEnd();
  }

  bool AtEnd() const {
    CHECK(impl_ != nullptr);
    return impl_->AtEnd();
  }

 private:
  std::unique_ptr<BatchIteratorImpl> impl_;
};

class BatchSet {
 public:
  explicit BatchSet(BatchIterator begin_iter) : begin_iter_(begin_iter) {}
  BatchIterator begin() { return begin_iter_; }
  BatchIterator end() { return BatchIterator(nullptr); }

 private:
  BatchIterator begin_iter_;
};

/*!
 * \brief This is data structure that user can pass to DMatrix::Create
 *  to create a DMatrix for training, user can create this data structure
 *  for customized Data Loading on single machine.
 *
 *  On distributed setting, usually an customized dmlc::Parser is needed instead.
 */
class DataSource : public dmlc::DataIter<SparsePage> {
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
class RowSet {
 public:
  /*! \return i-th row index */
  inline bst_uint operator[](size_t i) const;
  /*! \return the size of the set. */
  inline size_t Size() const;
  /*! \brief push the index back to the set */
  inline void PushBack(bst_uint i);
  /*! \brief clear the set */
  inline void Clear();
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
  RowSet()  = default;

 private:
  /*! \brief The internal data structure of size */
  uint64_t size_{0};
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
  DMatrix()  = default;
  /*! \brief meta information of the dataset */
  virtual MetaInfo& Info() = 0;
  /*! \brief meta information of the dataset */
  virtual const MetaInfo& Info() const = 0;
  /**
   * \brief Gets row batches. Use range based for loop over BatchSet to access individual batches.
   */
  virtual BatchSet GetRowBatches() = 0;
  virtual BatchSet GetSortedColumnBatches() = 0;
  virtual BatchSet GetColumnBatches() = 0;
  // the following are column meta data, should be able to answer them fast.
  /*! \return Whether the data columns single column block. */
  virtual bool SingleColBlock() const = 0;
  /*! \brief get column density */
  virtual float GetColDensity(size_t cidx) = 0;
  /*! \brief virtual destructor */
  virtual ~DMatrix() = default;
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
};

// implementation of inline functions
inline bst_uint RowSet::operator[](size_t i) const {
  return rows_.size() == 0 ? static_cast<bst_uint>(i) : rows_[i];
}

inline size_t RowSet::Size() const {
  return size_;
}

inline void RowSet::Clear() {
  rows_.clear(); size_ = 0;
}

inline void RowSet::PushBack(bst_uint i) {
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
DMLC_DECLARE_TRAITS(is_pod, xgboost::Entry, true);
DMLC_DECLARE_TRAITS(has_saveload, xgboost::RowSet, true);
}
#endif  // XGBOOST_DATA_H_
