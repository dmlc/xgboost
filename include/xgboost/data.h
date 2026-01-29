/**
 * Copyright 2015-2025, XGBoost Contributors
 * \file data.h
 * \brief The input data structure of xgboost.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_DATA_H_
#define XGBOOST_DATA_H_

#include <dmlc/base.h>
#include <dmlc/io.h>          // for Stream
#include <dmlc/serializer.h>  // for Handler
#include <xgboost/base.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/linalg.h>
#include <xgboost/span.h>
#include <xgboost/string_view.h>

#include <algorithm>
#include <cstdint>  // for int32_t, uint8_t
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace xgboost {
// forward declare dmatrix.
class DMatrix;
struct Context;

/*! \brief data type accepted by xgboost interface */
enum class DataType : uint8_t {
  kFloat32 = 1,
  kDouble = 2,
  kUInt32 = 3,
  kUInt64 = 4,
  kStr = 5
};

enum class FeatureType : uint8_t { kNumerical = 0, kCategorical = 1 };

enum class DataSplitMode : int { kRow = 0, kCol = 1 };

// Forward declaration of the container used by the meta info.
class CatContainer;

/**
 * @brief Meta information about dataset, always sit in memory.
 */
class MetaInfo {
 public:
  /*! \brief number of data fields in MetaInfo */
  static constexpr uint64_t kNumField = 13;

  /*! \brief number of rows in the data */
  bst_idx_t num_row_{0};  // NOLINT
  /*! \brief number of columns in the data */
  uint64_t num_col_{0};  // NOLINT
  /*! \brief number of nonzero entries in the data */
  uint64_t num_nonzero_{0};  // NOLINT
  /*! \brief label of each instance */
  linalg::Tensor<float, 2> labels;
  /*! \brief data split mode */
  DataSplitMode data_split_mode{DataSplitMode::kRow};
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
  linalg::Matrix<float> base_margin_;  // NOLINT
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

  MetaInfo();
  MetaInfo(MetaInfo&& that) = default;
  MetaInfo(MetaInfo const& that) = delete;
  MetaInfo& operator=(MetaInfo&& that) = default;
  MetaInfo& operator=(MetaInfo const& that) = delete;

  /**
   * @brief Validate all metainfo.
   */
  void Validate(DeviceOrd device) const;
  /**
   * @brief Slice the meta info.
   *
   * The device of ridxs is specified by the ctx object.
   *
   * @param ridxs Index of selected rows.
   * @param nnz   The number of non-missing values.
   */
  MetaInfo Slice(Context const* ctx, common::Span<bst_idx_t const> ridxs, bst_idx_t nnz) const;

  MetaInfo Copy() const;
  /**
   * @brief Whether the matrix is dense.
   */
  bool IsDense() const { return num_col_ * num_row_ == num_nonzero_; }
  /*!
   * \brief Get weight of each instances.
   * \param i Instance index.
   * \return The weight.
   */
  inline bst_float GetWeight(size_t i) const {
    return weights_.Size() != 0 ?  weights_.HostVector()[i] : 1.0f;
  }
  /*! \brief get sorted indexes (argsort) of labels by absolute value (used by cox loss) */
  const std::vector<size_t>& LabelAbsSort(Context const* ctx) const;
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
   * \brief Set information in the meta info with array interface.
   * \param key The key of the information.
   * \param interface_str String representation of json format array interface.
   */
  void SetInfo(Context const& ctx, StringView key, StringView interface_str);

  void GetInfo(char const* key, bst_ulong* out_len, DataType dtype,
               const void** out_dptr) const;

  void SetFeatureInfo(const char *key, const char **info, const bst_ulong size);
  void GetFeatureInfo(const char *field, std::vector<std::string>* out_str_vecs) const;

  /**
   * @brief Extend with other MetaInfo.
   *
   * @param that The other MetaInfo object.
   *
   * @param accumulate_rows Whether rows need to be accumulated in this function.  If
   *                        client code knows number of rows in advance, set this
   *                        parameter to false.
   * @param check_column Whether the extend method should check the consistency of
   *                     columns.
   */
  void Extend(MetaInfo const& that, bool accumulate_rows, bool check_column);
  /**
   * @brief Synchronize the number of columns across all workers.
   *
   * Normally we just need to find the maximum number of columns across all workers, but
   * in vertical federated learning, since each worker loads its own list of columns,
   * we need to sum them.
   */
  void SynchronizeNumberOfColumns(Context const* ctx, DataSplitMode split_mode);

  /** @brief Whether the data is split row-wise. */
  [[nodiscard]] bool IsRowSplit() const { return data_split_mode == DataSplitMode::kRow; }
  /** @brief Whether the data is split column-wise. */
  [[nodiscard]] bool IsColumnSplit() const { return data_split_mode == DataSplitMode::kCol; }
  /** @brief Whether this is a learning to rank data. */
  [[nodiscard]] bool IsRanking() const { return !group_ptr_.empty(); }

  /**
   * @brief A convenient method to check if we are doing vertical federated learning, which requires
   * some special processing.
   */
  [[nodiscard]] bool IsVerticalFederated() const;

  /*!
   * \brief A convenient method to check if the MetaInfo should contain labels.
   *
   * Normally we assume labels are available everywhere. The only exception is in vertical federated
   * learning where labels are only available on worker 0.
   */
  bool ShouldHaveLabels() const;
  /**
   * @brief Flag for whether the DMatrix has categorical features.
   */
  bool HasCategorical() const { return has_categorical_; }
  /**
   * @brief Getters for categories.
   */
  [[nodiscard]] CatContainer const* Cats() const;
  [[nodiscard]] CatContainer* Cats();
  [[nodiscard]] std::shared_ptr<CatContainer const>  CatsShared() const;
  /**
   * @brief Setter for categories.
   */
  void Cats(std::shared_ptr<CatContainer> cats);

 private:
  void SetInfoFromHost(Context const* ctx, StringView key, Json arr);
  void SetInfoFromCUDA(Context const* ctx, StringView key, Json arr);

  /*! \brief argsort of labels */
  mutable std::vector<size_t> label_order_cache_;
  bool has_categorical_{false};

  std::shared_ptr<CatContainer> cats_;
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

/**
 * @brief Parameters for constructing histogram index batches.
 */
struct BatchParam {
  /**
   * @brief Maximum number of bins per feature for histograms.
   */
  bst_bin_t max_bin{0};
  /**
   * @brief Hessian, used for sketching with future approx implementation.
   */
  common::Span<float const> hess;
  /**
   * @brief Whether should we force DMatrix to regenerate the batch.  Only used for
   *        GHistIndex.
   */
  bool regen{false};
  /**
   * @brief Forbid regenerating the gradient index. Used for internal validation.
   */
  bool forbid_regen{false};
  /**
   * @brief Parameter used to generate column matrix for hist.
   */
  double sparse_thresh{std::numeric_limits<double>::quiet_NaN()};
  /**
   * @brief Used for GPU external memory. Whether to copy the data into device.
   *
   * This affects only the current round of iteration.
   */
  bool prefetch_copy{true};
  /**
   * @brief The number of batches to pre-fetch for external memory.
   */
  std::int32_t n_prefetch_batches{3};
  /**
   * @brief Exact or others that don't need histogram.
   */
  BatchParam() = default;
  /**
   * @brief Used by the hist tree method.
   */
  BatchParam(bst_bin_t max_bin, double sparse_thresh)
      : max_bin{max_bin}, sparse_thresh{sparse_thresh} {}
  /**
   * @brief Used by the approx tree method.
   *
   *   Get batch with sketch weighted by hessian.  The batch will be regenerated if the
   *   span is changed, so caller should keep the span for each iteration.
   */
  BatchParam(bst_bin_t max_bin, common::Span<float const> hessian, bool regenerate)
      : max_bin{max_bin}, hess{hessian}, regen{regenerate} {}

  [[nodiscard]] bool ParamNotEqual(BatchParam const& other) const {
    // Check non-floating parameters.
    bool cond = max_bin != other.max_bin;
    // Check sparse thresh.
    bool l_nan = std::isnan(sparse_thresh);
    bool r_nan = std::isnan(other.sparse_thresh);
    bool st_chg = (l_nan != r_nan) || (!l_nan && !r_nan && (sparse_thresh != other.sparse_thresh));
    cond |= st_chg;

    return cond;
  }
  [[nodiscard]] bool Initialized() const { return max_bin != 0; }
  /**
   * @brief Make a copy of self for DMatrix to describe how its existing index was generated.
   */
  [[nodiscard]] BatchParam MakeCache() const {
    auto p = *this;
    // These parameters have nothing to do with how the gradient index was generated in the
    // first place.
    p.regen = false;
    p.forbid_regen = false;
    return p;
  }
};

struct HostSparsePageView {
  using Inst = common::Span<Entry const>;

  common::Span<bst_idx_t const> offset;
  common::Span<Entry const> data;

  [[nodiscard]] Inst operator[](std::size_t i) const {
    auto size = *(offset.data() + i + 1) - *(offset.data() + i);
    return {data.data() + *(offset.data() + i), static_cast<Inst::index_type>(size)};
  }

  [[nodiscard]] size_t Size() const { return offset.size() == 0 ? 0 : offset.size() - 1; }
};

/*!
 * \brief In-memory storage unit of sparse batch, stored in CSR format.
 */
class SparsePage {
 public:
  // Offset for each row.
  HostDeviceVector<bst_idx_t> offset;
  /*! \brief the data of the segments */
  HostDeviceVector<Entry> data;

  size_t base_rowid {0};

  /*! \brief an instance of sparse vector in the batch */
  using Inst = common::Span<Entry const>;

  [[nodiscard]] HostSparsePageView GetView() const {
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
  [[nodiscard]] size_t Size() const {
    return offset.Size() == 0 ? 0 : offset.Size() - 1;
  }

  /*! \return estimation of memory cost of this page */
  [[nodiscard]] size_t MemCostBytes() const {
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

  [[nodiscard]] SparsePage GetTranspose(int num_columns, int32_t n_threads) const;

  /**
   * \brief Sort the column index.
   */
  void SortIndices(int32_t n_threads);
  /**
   * \brief Check wether the column index is sorted.
   */
  [[nodiscard]] bool IsIndicesSorted(int32_t n_threads) const;
  /**
   * \brief Reindex the column index with an offset.
   */
  void Reindex(uint64_t feature_offset, int32_t n_threads);

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
  bst_idx_t Push(AdapterBatchT const& batch, float missing, std::int32_t nthread);

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

class EllpackPage;
class GHistIndexMatrix;

template<typename T>
class BatchIteratorImpl {
 public:
  using iterator_category = std::forward_iterator_tag;  // NOLINT
  virtual ~BatchIteratorImpl() = default;
  virtual const T& operator*() const = 0;
  virtual BatchIteratorImpl& operator++() = 0;
  [[nodiscard]] virtual bool AtEnd() const = 0;
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

  [[nodiscard]] bool operator!=(const BatchIterator&) const { return !this->AtEnd(); }

  [[nodiscard]] bool AtEnd() const {
    CHECK(impl_ != nullptr);
    return impl_->AtEnd();
  }

  [[nodiscard]] std::shared_ptr<T const> Page() const {
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

// Configuration for external memoroy DMatrix.
struct ExtMemConfig {
  // Cache prefix, not used if the cache is in the host memory. (on_host is true)
  std::string cache;
  // Whether the ellpack page is stored in the host memory.
  bool on_host;
  // Host cache/Total cache for the GPU impl.
  float cache_host_ratio;
  // Minimum number of of bytes for each ellpack page in cache. Only used for in-host
  // ExtMemQdm.
  std::int64_t min_cache_page_bytes;
  // Missing value.
  float missing;
  // The number of CPU threads.
  std::int32_t n_threads{0};
  // The ratio of the cache that can be compressed. Used for testing.
  float hw_decomp_ratio{std::numeric_limits<float>::quiet_NaN()};
  // Fallback to using nvcomp. Used for testing.
  bool allow_decomp_fallback{false};

  ExtMemConfig() = delete;
  ExtMemConfig(std::string cache, bool on_host, float h_ratio, std::int64_t min_cache,
               float missing, std::int32_t n_threads)
      : cache{std::move(cache)},
        on_host{on_host},
        cache_host_ratio{h_ratio},
        min_cache_page_bytes{min_cache},
        missing{missing},
        n_threads{n_threads} {}

  ExtMemConfig& SetParamsForTest(float _hw_decomp_ratio, bool _allow_decomp_fallback) {
    this->hw_decomp_ratio = _hw_decomp_ratio;
    this->allow_decomp_fallback = _allow_decomp_fallback;
    return *this;
  }
};

/**
 * @brief Internal data structured used by XGBoost to hold all external data.
 *
 *    There are multiple variants of the DMatrix class and can be accessed through the
 *    @ref Create() methods. The DMatrix itself holds the predictor `X`, and other data
 *    including labels and sample weights are stored in the @ref MetaInfo class.
 */
class DMatrix {
 public:
  /*! \brief default constructor */
  DMatrix()  = default;
  /** @brief meta information of the dataset */
  [[nodiscard]] virtual MetaInfo& Info() = 0;
  virtual void SetInfo(const char* key, std::string const& interface_str) {
    auto const& ctx = *this->Ctx();
    this->Info().SetInfo(ctx, key, StringView{interface_str});
  }
  /** @brief meta information of the dataset */
  [[nodiscard]] virtual const MetaInfo& Info() const = 0;

  /*! \brief Get thread local memory for returning data from DMatrix. */
  [[nodiscard]] XGBAPIThreadLocalEntry& GetThreadLocal() const;
  /**
   * @brief Get the context object of this DMatrix.  The context is created during construction of
   *        DMatrix with user specified `nthread` parameter.
   */
  [[nodiscard]] virtual Context const* Ctx() const = 0;

  /**
   * @brief Gets batches. Use range based for loop over BatchSet to access individual batches.
   */
  template <typename T>
  BatchSet<T> GetBatches();
  template <typename T>
  BatchSet<T> GetBatches(Context const* ctx);
  template <typename T>
  BatchSet<T> GetBatches(Context const* ctx, const BatchParam& param);
  template <typename T>
  [[nodiscard]] bool PageExists() const;

  /**
   * @return Whether the contains a single batch.
   *
   * The naming is legacy.
   */
  [[nodiscard]] bool SingleColBlock() const { return this->NumBatches() == 1; }
  [[nodiscard]] virtual std::int32_t NumBatches() const { return 1; }

  virtual ~DMatrix();

  /**
   * @brief Whether the matrix is dense.
   */
  [[nodiscard]] bool IsDense() const { return this->Info().IsDense(); }

  /**
   * @brief Load DMatrix from URI.
   *
   * @param uri The URI of input.
   * @param silent Whether print information during loading.
   * @param data_split_mode Indicate how the data was split beforehand.
   * @return The created DMatrix.
   */
  static DMatrix* Load(const std::string& uri, bool silent = true,
                       DataSplitMode data_split_mode = DataSplitMode::kRow);

  /**
   * @brief Creates a new DMatrix from an external data adapter.
   *
   * @tparam  AdapterT  Type of the adapter.
   * @param [in,out]  adapter         View onto an external data.
   * @param           missing         Values to count as missing.
   * @param           nthread         Number of threads for construction.
   * @param           cache_prefix    (Optional) The cache prefix for external memory.
   * @param           data_split_mode (Optional) Data split mode.
   *
   * @return  a Created DMatrix.
   */
  template <typename AdapterT>
  static DMatrix* Create(AdapterT* adapter, float missing, int nthread,
                         const std::string& cache_prefix = "",
                         DataSplitMode data_split_mode = DataSplitMode::kRow);

  /**
   * @brief Create a new Quantile based DMatrix used for histogram based algorithm.
   *
   * @tparam DataIterHandle         External iterator type, defined in C API.
   * @tparam DMatrixHandle          DMatrix handle, defined in C API.
   * @tparam DataIterResetCallback  Callback for reset, prototype defined in C API.
   * @tparam XGDMatrixCallbackNext  Callback for next, prototype defined in C API.
   *
   * @param iter    External data iterator
   * @param proxy   A hanlde to ProxyDMatrix
   * @param ref     Reference Quantile DMatrix.
   * @param reset   Callback for reset
   * @param next    Callback for next
   * @param missing Value that should be treated as missing.
   * @param nthread number of threads used for initialization.
   * @param max_bin Maximum number of bins.
   *
   * @return A created quantile based DMatrix.
   */
  template <typename DataIterHandle, typename DMatrixHandle, typename DataIterResetCallback,
            typename XGDMatrixCallbackNext>
  static DMatrix* Create(DataIterHandle iter, DMatrixHandle proxy, std::shared_ptr<DMatrix> ref,
                         DataIterResetCallback* reset, XGDMatrixCallbackNext* next, float missing,
                         std::int32_t nthread, bst_bin_t max_bin, std::int64_t max_quantile_blocks);

  /**
   * @brief Create an external memory DMatrix with callbacks.
   *
   * @tparam DataIterHandle         External iterator type, defined in C API.
   * @tparam DMatrixHandle          DMatrix handle, defined in C API.
   * @tparam DataIterResetCallback  Callback for reset, prototype defined in C API.
   * @tparam XGDMatrixCallbackNext  Callback for next, prototype defined in C API.
   *
   * @param iter    External data iterator
   * @param proxy   A hanlde to ProxyDMatrix
   * @param reset   Callback for reset
   * @param next    Callback for next
   * @param config  Configuration for the cache.
   *
   * @return A created external memory DMatrix.
   */
  template <typename DataIterHandle, typename DMatrixHandle, typename DataIterResetCallback,
            typename XGDMatrixCallbackNext>
  static DMatrix* Create(DataIterHandle iter, DMatrixHandle proxy, DataIterResetCallback* reset,
                         XGDMatrixCallbackNext* next, ExtMemConfig const& config);

  /**
   * @brief Create an external memory quantile DMatrix with callbacks.
   *
   *     Parameters are a combination of the external memory DMatrix and the quantile DMatrix.
   *
   * @return A created external memory quantile DMatrix.
   */
  template <typename DataIterHandle, typename DMatrixHandle, typename DataIterResetCallback,
            typename XGDMatrixCallbackNext>
  static DMatrix* Create(DataIterHandle iter, DMatrixHandle proxy, std::shared_ptr<DMatrix> ref,
                         DataIterResetCallback* reset, XGDMatrixCallbackNext* next,
                         bst_bin_t max_bin, std::int64_t max_quantile_blocks,
                         ExtMemConfig const& config);

  virtual DMatrix *Slice(common::Span<int32_t const> ridxs) = 0;

  /**
   * @brief Slice a DMatrix by columns.
   *
   * @param num_slices Total number of slices
   * @param slice_id Index of the current slice
   * @return DMatrix containing the slice of columns
   */
  virtual DMatrix* SliceCol(int num_slices, int slice_id) = 0;
  /**
   * @brief Accessor for the string representation of the categories.
   */
  [[nodiscard]] CatContainer const* Cats() const { return this->CatsShared().get(); }
  [[nodiscard]] std::shared_ptr<CatContainer const> CatsShared() const {
    return this->Info().CatsShared();
  }

 protected:
  virtual BatchSet<SparsePage> GetRowBatches() = 0;
  virtual BatchSet<CSCPage> GetColumnBatches(Context const* ctx) = 0;
  virtual BatchSet<SortedCSCPage> GetSortedColumnBatches(Context const* ctx) = 0;
  virtual BatchSet<EllpackPage> GetEllpackBatches(Context const* ctx, BatchParam const& param) = 0;
  virtual BatchSet<GHistIndexMatrix> GetGradientIndex(Context const* ctx,
                                                      BatchParam const& param) = 0;
  virtual BatchSet<ExtSparsePage> GetExtBatches(Context const* ctx, BatchParam const& param) = 0;

  [[nodiscard]] virtual bool EllpackExists() const = 0;
  [[nodiscard]] virtual bool GHistIndexExists() const = 0;
  [[nodiscard]] virtual bool SparsePageExists() const = 0;
};

template <>
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

template <>
inline bool DMatrix::PageExists<SparsePage>() const {
  return this->SparsePageExists();
}

template <>
inline BatchSet<SparsePage> DMatrix::GetBatches(Context const*) {
  return GetRowBatches();
}

template <>
inline BatchSet<CSCPage> DMatrix::GetBatches(Context const* ctx) {
  return GetColumnBatches(ctx);
}

template <>
inline BatchSet<SortedCSCPage> DMatrix::GetBatches(Context const* ctx) {
  return GetSortedColumnBatches(ctx);
}

template <>
inline BatchSet<EllpackPage> DMatrix::GetBatches(Context const* ctx, BatchParam const& param) {
  return GetEllpackBatches(ctx, param);
}

template <>
inline BatchSet<GHistIndexMatrix> DMatrix::GetBatches(Context const* ctx, BatchParam const& param) {
  return GetGradientIndex(ctx, param);
}

template <>
inline BatchSet<ExtSparsePage> DMatrix::GetBatches(Context const* ctx, BatchParam const& param) {
  return GetExtBatches(ctx, param);
}
}  // namespace xgboost

DECLARE_FIELD_ENUM_CLASS(xgboost::DataSplitMode);

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
