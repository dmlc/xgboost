/*!
 * Copyright (c) 2015-2022 by XGBoost Contributors
 * \file data.h
 * \brief The input data structure of xgboost.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_DATA_H_
#define XGBOOST_DATA_H_

#include <dmlc/base.h>
#include <dmlc/data.h>
#include <dmlc/serializer.h>
#include <xgboost/base.h>
#include <xgboost/generic_parameters.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/linalg.h>
#include <xgboost/span.h>
#include <xgboost/string_view.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace xgboost {
// forward declare dmatrix.
class DMatrix;

/*! \brief data type accepted by xgboost interface */
enum class DataType : uint8_t {
  kFloat32 = 1,
  kDouble = 2,
  kUInt32 = 3,
  kUInt64 = 4,
  kStr = 5
};

enum class FeatureType : uint8_t { kNumerical = 0, kCategorical = 1 };

/*!
 * \brief Meta information about dataset, always sit in memory.
 */
class MetaInfo {
 public:
  /*! \brief number of data fields in MetaInfo */
  static constexpr uint64_t kNumField = 12;

  /*! \brief number of rows in the data */
  uint64_t num_row_{0};  // NOLINT
  /*! \brief number of columns in the data */
  uint64_t num_col_{0};  // NOLINT
  /*! \brief number of nonzero entries in the data */
  uint64_t num_nonzero_{0};  // NOLINT
  /*! \brief label of each instance */
  linalg::Tensor<float, 2> labels;
  /*!
   * \brief the index of begin and end of a group
   *  needed when the learning task is ranking.
   */
  std::vector<bst_group_t> group_ptr_;  // NOLINT
  /*! \brief weights of each instance, optional */
  HostDeviceVector<bst_float> weights_;  // NOLINT
  /*!
   * \brief initialized margins,
   * if specified, xgboost will start from this init margin
   * can be used to specify initial prediction to boost from.
   */
  linalg::Tensor<float, 2> base_margin_;  // NOLINT
  /*!
   * \brief lower bound of the label, to be used for survival analysis (censored regression)
   */
  HostDeviceVector<bst_float> labels_lower_bound_;  // NOLINT
  /*!
   * \brief upper bound of the label, to be used for survival analysis (censored regression)
   */
  HostDeviceVector<bst_float> labels_upper_bound_;  // NOLINT

  /*!
   * \brief Name of type for each feature provided by users. Eg. "int"/"float"/"i"/"q"
   */
  std::vector<std::string> feature_type_names;
  /*!
   * \brief Name for each feature.
   */
  std::vector<std::string> feature_names;
  /*
   * \brief Type of each feature.  Automatically set when feature_type_names is specifed.
   */
  HostDeviceVector<FeatureType> feature_types;
  /*
   * \brief Weight of each feature, used to define the probability of each feature being
   *        selected when using column sampling.
   */
  HostDeviceVector<float> feature_weights;

  /*! \brief default constructor */
  MetaInfo()  = default;
  MetaInfo(MetaInfo&& that) = default;
  MetaInfo& operator=(MetaInfo&& that) = default;
  MetaInfo& operator=(MetaInfo const& that) = delete;

  /*!
   * \brief Validate all metainfo.
   */
  void Validate(int32_t device) const;

  MetaInfo Slice(common::Span<int32_t const> ridxs) const;
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
    if (label_order_cache_.size() == labels.Size()) {
      return label_order_cache_;
    }
    label_order_cache_.resize(labels.Size());
    std::iota(label_order_cache_.begin(), label_order_cache_.end(), 0);
    const auto& l = labels.Data()->HostVector();
    XGBOOST_PARALLEL_STABLE_SORT(label_order_cache_.begin(), label_order_cache_.end(),
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
  void SetInfo(Context const& ctx, const char* key, const void* dptr, DataType dtype, size_t num);
  /*!
   * \brief Set information in the meta info with array interface.
   * \param key The key of the information.
   * \param interface_str String representation of json format array interface.
   */
  void SetInfo(Context const& ctx, StringView key, StringView interface_str);

  void GetInfo(char const* key, bst_ulong* out_len, DataType dtype,
               const void** out_dptr) const;

  void SetFeatureInfo(const char *key, const char **info, const bst_ulong size);
  void GetFeatureInfo(const char *field, std::vector<std::string>* out_str_vecs) const;

  /*
   * \brief Extend with other MetaInfo.
   *
   * \param that The other MetaInfo object.
   *
   * \param accumulate_rows Whether rows need to be accumulated in this function.  If
   *                        client code knows number of rows in advance, set this
   *                        parameter to false.
   * \param check_column Whether the extend method should check the consistency of
   *                     columns.
   */
  void Extend(MetaInfo const& that, bool accumulate_rows, bool check_column);

 private:
  void SetInfoFromHost(Context const& ctx, StringView key, Json arr);
  void SetInfoFromCUDA(Context const& ctx, StringView key, Json arr);

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
  XGBOOST_DEVICE Entry(bst_feature_t index, bst_float fvalue) : index(index), fvalue(fvalue) {}
  /*! \brief reversely compare feature values */
  inline static bool CmpValue(const Entry& a, const Entry& b) {
    return a.fvalue < b.fvalue;
  }
  static bool CmpIndex(Entry const& a, Entry const& b) {
    return a.index < b.index;
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
  int gpu_id {-1};
  /*! \brief Maximum number of bins per feature for histograms. */
  bst_bin_t max_bin{0};
  /*! \brief Hessian, used for sketching with future approx implementation. */
  common::Span<float> hess;
  /*! \brief Whether should DMatrix regenerate the batch.  Only used for GHistIndex. */
  bool regen {false};
  /*! \brief Parameter used to generate column matrix for hist. */
  double sparse_thresh{std::numeric_limits<double>::quiet_NaN()};

  BatchParam() = default;
  // GPU Hist
  BatchParam(int32_t device, bst_bin_t max_bin)
      : gpu_id{device}, max_bin{max_bin} {}
  // Hist
  BatchParam(bst_bin_t max_bin, double sparse_thresh)
      : max_bin{max_bin}, sparse_thresh{sparse_thresh} {}
  // Approx
  /**
   * \brief Get batch with sketch weighted by hessian.  The batch will be regenerated if
   *        the span is changed, so caller should keep the span for each iteration.
   */
  BatchParam(bst_bin_t max_bin, common::Span<float> hessian, bool regenerate)
      : max_bin{max_bin}, hess{hessian}, regen{regenerate} {}

  bool operator!=(BatchParam const& other) const {
    if (hess.empty() && other.hess.empty()) {
      return gpu_id != other.gpu_id || max_bin != other.max_bin;
    }
    return gpu_id != other.gpu_id || max_bin != other.max_bin || hess.data() != other.hess.data();
  }
  bool operator==(BatchParam const& other) const {
    return !(*this != other);
  }
};

struct HostSparsePageView {
  using Inst = common::Span<Entry const>;

  common::Span<bst_row_t const> offset;
  common::Span<Entry const> data;

  Inst operator[](size_t i) const {
    auto size = *(offset.data() + i + 1) - *(offset.data() + i);
    return {data.data() + *(offset.data() + i),
            static_cast<Inst::index_type>(size)};
  }

  size_t Size() const { return offset.size() == 0 ? 0 : offset.size() - 1; }
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

  size_t base_rowid {0};

  /*! \brief an instance of sparse vector in the batch */
  using Inst = common::Span<Entry const>;

  HostSparsePageView GetView() const {
    return {offset.ConstHostSpan(), data.ConstHostSpan()};
  }

  /*! \brief constructor */
  SparsePage() {
    this->Clear();
  }

  SparsePage(SparsePage const& that) = delete;
  SparsePage(SparsePage&& that) = default;
  SparsePage& operator=(SparsePage const& that) = delete;
  SparsePage& operator=(SparsePage&& that) = default;
  virtual ~SparsePage() = default;

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

  SparsePage GetTranspose(int num_columns, int32_t n_threads) const;

  /**
   * \brief Sort the column index.
   */
  void SortIndices(int32_t n_threads);
  /**
   * \brief Check wether the column index is sorted.
   */
  bool IsIndicesSorted(int32_t n_threads) const;

  void SortRows(int32_t n_threads);

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

/**
 * \brief Sparse page for exporting DMatrix. Same as SparsePage, just a different type to
 *        prevent being used internally.
 */
class ExtSparsePage {
 public:
  std::shared_ptr<SparsePage const> page;
  explicit ExtSparsePage(std::shared_ptr<SparsePage const> p) : page{std::move(p)} {}
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

  EllpackPage(EllpackPage&& that);

  /*! \return Number of instances in the page. */
  size_t Size() const;

  /*! \brief Set the base row id for this page. */
  void SetBaseRowId(size_t row_id);

  const EllpackPageImpl* Impl() const { return impl_.get(); }
  EllpackPageImpl* Impl() { return impl_.get(); }

 private:
  std::unique_ptr<EllpackPageImpl> impl_;
};

class GHistIndexMatrix;

template<typename T>
class BatchIteratorImpl {
 public:
  using iterator_category = std::forward_iterator_tag;  // NOLINT
  virtual ~BatchIteratorImpl() = default;
  virtual const T& operator*() const = 0;
  virtual BatchIteratorImpl& operator++() = 0;
  virtual bool AtEnd() const = 0;
  virtual std::shared_ptr<T const> Page() const = 0;
};

template<typename T>
class BatchIterator {
 public:
  using iterator_category = std::forward_iterator_tag;  // NOLINT
  explicit BatchIterator(BatchIteratorImpl<T>* impl) { impl_.reset(impl); }
  explicit BatchIterator(std::shared_ptr<BatchIteratorImpl<T>> impl) { impl_ = impl; }

  BatchIterator &operator++() {
    CHECK(impl_ != nullptr);
    ++(*impl_);
    return *this;
  }

  const T& operator*() const {
    CHECK(impl_ != nullptr);
    return *(*impl_);
  }

  bool operator!=(const BatchIterator&) const {
    CHECK(impl_ != nullptr);
    return !impl_->AtEnd();
  }

  bool AtEnd() const {
    CHECK(impl_ != nullptr);
    return impl_->AtEnd();
  }

  std::shared_ptr<T const> Page() const {
    return impl_->Page();
  }

 private:
  std::shared_ptr<BatchIteratorImpl<T>> impl_;
};

template<typename T>
class BatchSet {
 public:
  explicit BatchSet(BatchIterator<T> begin_iter) : begin_iter_(std::move(begin_iter)) {}
  BatchIterator<T> begin() { return begin_iter_; }  // NOLINT
  BatchIterator<T> end() { return BatchIterator<T>(nullptr); }  // NOLINT

 private:
  BatchIterator<T> begin_iter_;
};

struct XGBAPIThreadLocalEntry;

/*!
 * \brief Internal data structured used by XGBoost during training.
 */
class DMatrix {
 public:
  /*! \brief default constructor */
  DMatrix()  = default;
  /*! \brief meta information of the dataset */
  virtual MetaInfo& Info() = 0;
  virtual void SetInfo(const char* key, const void* dptr, DataType dtype, size_t num) {
    auto const& ctx = *this->Ctx();
    this->Info().SetInfo(ctx, key, dptr, dtype, num);
  }
  virtual void SetInfo(const char* key, std::string const& interface_str) {
    auto const& ctx = *this->Ctx();
    this->Info().SetInfo(ctx, key, StringView{interface_str});
  }
  /*! \brief meta information of the dataset */
  virtual const MetaInfo& Info() const = 0;

  /*! \brief Get thread local memory for returning data from DMatrix. */
  XGBAPIThreadLocalEntry& GetThreadLocal() const;
  /**
   * \brief Get the context object of this DMatrix.  The context is created during construction of
   *        DMatrix with user specified `nthread` parameter.
   */
  virtual Context const* Ctx() const = 0;

  /**
   * \brief Gets batches. Use range based for loop over BatchSet to access individual batches.
   */
  template <typename T>
  BatchSet<T> GetBatches();
  template <typename T>
  BatchSet<T> GetBatches(const BatchParam& param);
  template <typename T>
  bool PageExists() const;

  // the following are column meta data, should be able to answer them fast.
  /*! \return Whether the data columns single column block. */
  virtual bool SingleColBlock() const = 0;
  /*! \brief virtual destructor */
  virtual ~DMatrix();

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
                       const std::string& file_format = "auto");

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
                         const std::string& cache_prefix = "");

  /**
   * \brief Create a new Quantile based DMatrix used for histogram based algorithm.
   *
   * \tparam DataIterHandle         External iterator type, defined in C API.
   * \tparam DMatrixHandle          DMatrix handle, defined in C API.
   * \tparam DataIterResetCallback  Callback for reset, prototype defined in C API.
   * \tparam XGDMatrixCallbackNext  Callback for next, prototype defined in C API.
   *
   * \param iter    External data iterator
   * \param proxy   A hanlde to ProxyDMatrix
   * \param ref     Reference Quantile DMatrix.
   * \param reset   Callback for reset
   * \param next    Callback for next
   * \param missing Value that should be treated as missing.
   * \param nthread number of threads used for initialization.
   * \param max_bin Maximum number of bins.
   *
   * \return A created quantile based DMatrix.
   */
  template <typename DataIterHandle, typename DMatrixHandle, typename DataIterResetCallback,
            typename XGDMatrixCallbackNext>
  static DMatrix* Create(DataIterHandle iter, DMatrixHandle proxy, std::shared_ptr<DMatrix> ref,
                         DataIterResetCallback* reset, XGDMatrixCallbackNext* next, float missing,
                         int nthread, bst_bin_t max_bin);

  /**
   * \brief Create an external memory DMatrix with callbacks.
   *
   * \tparam DataIterHandle         External iterator type, defined in C API.
   * \tparam DMatrixHandle          DMatrix handle, defined in C API.
   * \tparam DataIterResetCallback  Callback for reset, prototype defined in C API.
   * \tparam XGDMatrixCallbackNext  Callback for next, prototype defined in C API.
   *
   * \param iter    External data iterator
   * \param proxy   A hanlde to ProxyDMatrix
   * \param reset   Callback for reset
   * \param next    Callback for next
   * \param missing Value that should be treated as missing.
   * \param nthread number of threads used for initialization.
   * \param cache   Prefix of cache file path.
   *
   * \return A created external memory DMatrix.
   */
  template <typename DataIterHandle, typename DMatrixHandle,
            typename DataIterResetCallback, typename XGDMatrixCallbackNext>
  static DMatrix *Create(DataIterHandle iter, DMatrixHandle proxy,
                         DataIterResetCallback *reset,
                         XGDMatrixCallbackNext *next, float missing,
                         int32_t nthread, std::string cache);

  virtual DMatrix *Slice(common::Span<int32_t const> ridxs) = 0;
  /*! \brief Number of rows per page in external memory.  Approximately 100MB per page for
   *  dataset with 100 features. */
  static const size_t kPageSize = 32UL << 12UL;

 protected:
  virtual BatchSet<SparsePage> GetRowBatches() = 0;
  virtual BatchSet<CSCPage> GetColumnBatches() = 0;
  virtual BatchSet<SortedCSCPage> GetSortedColumnBatches() = 0;
  virtual BatchSet<EllpackPage> GetEllpackBatches(const BatchParam& param) = 0;
  virtual BatchSet<GHistIndexMatrix> GetGradientIndex(const BatchParam& param) = 0;
  virtual BatchSet<ExtSparsePage> GetExtBatches(BatchParam const& param) = 0;

  virtual bool EllpackExists() const = 0;
  virtual bool GHistIndexExists() const = 0;
  virtual bool SparsePageExists() const = 0;
};

template<>
inline BatchSet<SparsePage> DMatrix::GetBatches() {
  return GetRowBatches();
}

template <>
inline bool DMatrix::PageExists<EllpackPage>() const {
  return this->EllpackExists();
}

template <>
inline bool DMatrix::PageExists<GHistIndexMatrix>() const {
  return this->GHistIndexExists();
}

template<>
inline bool DMatrix::PageExists<SparsePage>() const {
  return this->SparsePageExists();
}

template<>
inline BatchSet<CSCPage> DMatrix::GetBatches() {
  return GetColumnBatches();
}

template<>
inline BatchSet<SortedCSCPage> DMatrix::GetBatches() {
  return GetSortedColumnBatches();
}

template<>
inline BatchSet<EllpackPage> DMatrix::GetBatches(const BatchParam& param) {
  return GetEllpackBatches(param);
}

template <>
inline BatchSet<GHistIndexMatrix> DMatrix::GetBatches(const BatchParam& param) {
  return GetGradientIndex(param);
}

template <>
inline BatchSet<ExtSparsePage> DMatrix::GetBatches() {
  return GetExtBatches(BatchParam{});
}
}  // namespace xgboost

namespace dmlc {
DMLC_DECLARE_TRAITS(is_pod, xgboost::Entry, true);

namespace serializer {

template <>
struct Handler<xgboost::Entry> {
  inline static void Write(Stream* strm, const xgboost::Entry& data) {
    strm->Write(data.index);
    strm->Write(data.fvalue);
  }

  inline static bool Read(Stream* strm, xgboost::Entry* data) {
    return strm->Read(&data->index) && strm->Read(&data->fvalue);
  }
};

}  // namespace serializer
}  // namespace dmlc
#endif  // XGBOOST_DATA_H_
