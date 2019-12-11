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
#include <rabit/rabit.h>
#include <xgboost/base.h>
#include <xgboost/span.h>
#include <xgboost/host_device_vector.h>

#include <memory>
#include <numeric>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace xgboost {
// forward declare dmatrix.
class DMatrix;

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
   * \brief the index of begin and end of a group
   *  needed when the learning task is ranking.
   */
  std::vector<bst_group_t> group_ptr_;
  /*! \brief weights of each instance, optional */
  HostDeviceVector<bst_float> weights_;
  /*!
   * \brief initialized margins,
   * if specified, xgboost will start from this init margin
   * can be used to specify initial prediction to boost from.
   */
  HostDeviceVector<bst_float> base_margin_;
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
  /*!
   * \brief Set information in the meta info with array interface.
   * \param key The key of the information.
   * \param interface_str String representation of json format array interface.
   *
   *          [ column_0, column_1, ... column_n ]
   *
   *        Right now only 1 column is permitted.
   */
  void SetInfo(const char* key, std::string const& interface_str);

 private:
  /*! \brief argsort of labels */
  mutable std::vector<size_t> label_order_cache_;
};

/*! \brief Element from a sparse vector */
struct Entry {
  /*! \brief feature index */
  bst_feature_t index;
  /*! \brief feature value */
  bst_float fvalue;
  /*! \brief default constructor */
  Entry() = default;
  /*!
   * \brief constructor with index and value
   * \param index The feature or row index.
   * \param fvalue The feature value.
   */
  Entry(bst_feature_t index, bst_float fvalue) : index(index), fvalue(fvalue) {}
  /*! \brief reversely compare feature values */
  inline static bool CmpValue(const Entry& a, const Entry& b) {
    return a.fvalue < b.fvalue;
  }
  inline bool operator==(const Entry& other) const {
    return (this->index == other.index && this->fvalue == other.fvalue);
  }
};

/*!
 * \brief Parameters for constructing batches.
 */
struct BatchParam {
  /*! \brief The GPU device to use. */
  int gpu_id;
  /*! \brief Maximum number of bins per feature for histograms. */
  int max_bin;
  /*! \brief Number of rows in a GPU batch, used for finding quantiles on GPU. */
  int gpu_batch_nrows;
  /*! \brief Page size for external memory mode. */
  size_t gpu_page_size;

  inline bool operator!=(const BatchParam& other) const {
    return gpu_id != other.gpu_id ||
        max_bin != other.max_bin ||
        gpu_batch_nrows != other.gpu_batch_nrows ||
        gpu_page_size != other.gpu_page_size;
  }
};

/*!
 * \brief In-memory storage unit of sparse batch, stored in CSR format.
 */
class SparsePage {
 public:
  // Offset for each row.
  HostDeviceVector<bst_row_t> offset;
  /*! \brief the data of the segments */
  HostDeviceVector<Entry> data;

  size_t base_rowid{};

  /*! \brief an instance of sparse vector in the batch */
  using Inst = common::Span<Entry const>;

  /*! \brief get i-th row from the batch */
  inline Inst operator[](size_t i) const {
    const auto& data_vec = data.HostVector();
    const auto& offset_vec = offset.HostVector();
    size_t size;
    // in distributed mode, some partitions may not get any instance for a feature. Therefore
    // we should set the size as zero
    if (rabit::IsDistributed() && i + 1 >= offset_vec.size()) {
      size = 0;
    } else {
      size = offset_vec[i + 1] - offset_vec[i];
    }
    return {data_vec.data() + offset_vec[i],
            static_cast<Inst::index_type>(size)};
  }

  /*! \brief constructor */
  SparsePage() {
    this->Clear();
  }

  /*! \return Number of instances in the page. */
  inline size_t Size() const {
    return offset.Size() == 0 ? 0 : offset.Size() - 1;
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

  /*! \brief Set the base row id for this page. */
  inline void SetBaseRowId(size_t row_id) {
    base_rowid = row_id;
  }

  SparsePage GetTranspose(int num_columns) const;

  void SortRows() {
    auto ncol = static_cast<bst_omp_uint>(this->Size());
#pragma omp parallel for default(none) shared(ncol) schedule(dynamic, 1)
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
  void Push(const dmlc::RowBlock<uint32_t>& batch);

  /**
   * \brief Pushes external data batch onto this page
   *
   * \tparam  AdapterBatchT 
   * \param batch 
   * \param missing 
   * \param nthread 
   *
   * \return  The maximum number of columns encountered in this input batch. Useful when pushing many adapter batches to work out the total number of columns.
   */
  template <typename AdapterBatchT>
  uint64_t Push(const AdapterBatchT& batch, float missing, int nthread);

  /*!
   * \brief Push a sparse page
   * \param batch the row page
   */
  void Push(const SparsePage &batch);
  /*!
   * \brief Push a SparsePage stored in CSC format
   * \param batch The row batch to be pushed
   */
  void PushCSC(const SparsePage& batch);
};

class CSCPage: public SparsePage {
 public:
  CSCPage() : SparsePage() {}
  explicit CSCPage(SparsePage page) : SparsePage(std::move(page)) {}
};

class SortedCSCPage : public SparsePage {
 public:
  SortedCSCPage() : SparsePage() {}
  explicit SortedCSCPage(SparsePage page) : SparsePage(std::move(page)) {}
};

class EllpackPageImpl;
/*!
 * \brief A page stored in ELLPACK format.
 *
 * This class uses the PImpl idiom (https://en.cppreference.com/w/cpp/language/pimpl) to avoid
 * including CUDA-specific implementation details in the header.
 */
class EllpackPage {
 public:
  /*!
   * \brief Default constructor.
   *
   * This is used in the external memory case. An empty ELLPACK page is constructed with its content
   * set later by the reader.
   */
  EllpackPage();

  /*!
   * \brief Constructor from an existing DMatrix.
   *
   * This is used in the in-memory case. The ELLPACK page is constructed from an existing DMatrix
   * in CSR format.
   */
  explicit EllpackPage(DMatrix* dmat, const BatchParam& param);

  /*! \brief Destructor. */
  ~EllpackPage();

  /*! \return Number of instances in the page. */
  size_t Size() const;

  /*! \brief Set the base row id for this page. */
  void SetBaseRowId(size_t row_id);

  const EllpackPageImpl* Impl() const { return impl_.get(); }
  EllpackPageImpl* Impl() { return impl_.get(); }

 private:
  std::unique_ptr<EllpackPageImpl> impl_;
};

template<typename T>
class BatchIteratorImpl {
 public:
  virtual ~BatchIteratorImpl() = default;
  virtual T& operator*() = 0;
  virtual const T& operator*() const = 0;
  virtual void operator++() = 0;
  virtual bool AtEnd() const = 0;
};

template<typename T>
class BatchIterator {
 public:
  using iterator_category = std::forward_iterator_tag;
  explicit BatchIterator(BatchIteratorImpl<T>* impl) { impl_.reset(impl); }

  void operator++() {
    CHECK(impl_ != nullptr);
    ++(*impl_);
  }

  T& operator*() {
    CHECK(impl_ != nullptr);
    return *(*impl_);
  }

  const T& operator*() const {
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
  std::shared_ptr<BatchIteratorImpl<T>> impl_;
};

template<typename T>
class BatchSet {
 public:
  explicit BatchSet(BatchIterator<T> begin_iter) : begin_iter_(begin_iter) {}
  BatchIterator<T> begin() { return begin_iter_; }
  BatchIterator<T> end() { return BatchIterator<T>(nullptr); }

 private:
  BatchIterator<T> begin_iter_;
};

/*!
 * \brief This is data structure that user can pass to DMatrix::Create
 *  to create a DMatrix for training, user can create this data structure
 *  for customized Data Loading on single machine.
 *
 *  On distributed setting, usually an customized dmlc::Parser is needed instead.
 */
template<typename T>
class DataSource : public dmlc::DataIter<T> {
 public:
  /*!
   * \brief Meta information about the dataset
   * The subclass need to be able to load this correctly from data.
   */
  MetaInfo info;
};

/*!
 * \brief Internal data structured used by XGBoost during training.
 *  There are two ways to create a customized DMatrix that reads in user defined-format.
 *
 *  - Provide a dmlc::Parser and pass into the DMatrix::Create
 *  - Alternatively, if data can be represented by an URL, define a new dmlc::Parser and register by
 *    DMLC_REGISTER_DATA_PARSER;
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
   * \brief Gets batches. Use range based for loop over BatchSet to access individual batches.
   */
  template<typename T>
  BatchSet<T> GetBatches(const BatchParam& param = {});
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

  /*! \brief Whether the matrix is dense. */
  bool IsDense() const {
    return Info().num_nonzero_ == Info().num_row_ * Info().num_col_;
  }

  /*!
   * \brief Load DMatrix from URI.
   * \param uri The URI of input.
   * \param silent Whether print information during loading.
   * \param load_row_split Flag to read in part of rows, divided among the workers in distributed mode.
   * \param file_format The format type of the file, used for dmlc::Parser::Create.
   *   By default "auto" will be able to load in both local binary file.
   * \param page_size Page size for external memory.
   * \return The created DMatrix.
   */
  static DMatrix* Load(const std::string& uri,
                       bool silent,
                       bool load_row_split,
                       const std::string& file_format = "auto",
                       size_t page_size = kPageSize);

  /*!
   * \brief create a new DMatrix, by wrapping a row_iterator, and meta info.
   * \param source The source iterator of the data, the create function takes ownership of the source.
   * \param cache_prefix The path to prefix of temporary cache file of the DMatrix when used in external memory mode.
   *     This can be nullptr for common cases, and in-memory mode will be used.
   * \return a Created DMatrix.
   */
  static DMatrix* Create(std::unique_ptr<DataSource<SparsePage>>&& source,
                         const std::string& cache_prefix = "");

  /**
   * \brief Creates a new DMatrix from an external data adapter.
   *
   * \tparam  AdapterT  Type of the adapter.
   * \param [in,out]  adapter       View onto an external data.
   * \param           missing       Values to count as missing.
   * \param           nthread       Number of threads for construction.
   * \param           cache_prefix  (Optional) The cache prefix for external memory.
   * \param           page_size     (Optional) Size of the page.
   *
   * \return  a Created DMatrix.
   */
  template <typename AdapterT>
  static DMatrix* Create(AdapterT* adapter, float missing, int nthread,
                         const std::string& cache_prefix = "",
                         size_t page_size = kPageSize);


  /*! \brief page size 32 MB */
  static const size_t kPageSize = 32UL << 20UL;

 protected:
  virtual BatchSet<SparsePage> GetRowBatches() = 0;
  virtual BatchSet<CSCPage> GetColumnBatches() = 0;
  virtual BatchSet<SortedCSCPage> GetSortedColumnBatches() = 0;
  virtual BatchSet<EllpackPage> GetEllpackBatches(const BatchParam& param) = 0;
};

template<>
inline BatchSet<SparsePage> DMatrix::GetBatches(const BatchParam&) {
  return GetRowBatches();
}

template<>
inline BatchSet<CSCPage> DMatrix::GetBatches(const BatchParam&) {
  return GetColumnBatches();
}

template<>
inline BatchSet<SortedCSCPage> DMatrix::GetBatches(const BatchParam&) {
  return GetSortedColumnBatches();
}

template<>
inline BatchSet<EllpackPage> DMatrix::GetBatches(const BatchParam& param) {
  return GetEllpackBatches(param);
}
}  // namespace xgboost

namespace dmlc {
DMLC_DECLARE_TRAITS(is_pod, xgboost::Entry, true);
}
#endif  // XGBOOST_DATA_H_
