/**
 * Copyright 2020-2025, XGBoost contributors
 */
#ifndef XGBOOST_DATA_PROXY_DMATRIX_H_
#define XGBOOST_DATA_PROXY_DMATRIX_H_

#include <algorithm>    // for none_of
#include <any>          // for any, any_cast
#include <cstdint>      // for uint32_t, int32_t
#include <memory>       // for shared_ptr
#include <type_traits>  // for invoke_result_t, declval
#include <utility>      // for forward
#include <vector>       // for vector

#include "../common/nvtx_utils.h"  // for xgboost_NVTX_FN_RANGE
#include "../encoder/ordinal.h"    // for HostColumnsView
#include "adapter.h"               // for ColumnarAdapter, ArrayAdapter, MakeEncColumnarBatch
#include "cat_container.h"         // for CatContainer
#include "xgboost/c_api.h"         // for DataIterHandle
#include "xgboost/context.h"       // for Context
#include "xgboost/data.h"          // for MetaInfo
#include "xgboost/string_view.h"   // for StringView

namespace xgboost::data {
/**
 * @brief A proxy to external iterator.
 *
 * @note The external iterator is actually 1-based since the first call to @ref Next
 * increases the counter to 1 and it's necessary to call the @ref Next method at least
 * once to get data. We here along with the page source together convert it back to
 * 0-based by calling @ref Next in the page source's constructor.
 */
template <typename ResetFn, typename NextFn>
class DataIterProxy {
  DataIterHandle iter_;
  ResetFn* reset_;
  NextFn* next_;
  std::int32_t count_{0};

 public:
  DataIterProxy(DataIterHandle iter, ResetFn* reset, NextFn* next)
      : iter_{iter}, reset_{reset}, next_{next} {}
  DataIterProxy(DataIterProxy&& that) = default;
  DataIterProxy& operator=(DataIterProxy&& that) = default;
  DataIterProxy(DataIterProxy const& that) = delete;
  DataIterProxy& operator=(DataIterProxy const& that) = delete;

  [[nodiscard]] bool Next() {
    xgboost_NVTX_FN_RANGE();

    bool ret = !!next_(iter_);
    if (!ret) {
      return ret;
    }
    count_++;
    return ret;
  }
  void Reset() {
    reset_(iter_);
    count_ = 0;
  }
  [[nodiscard]] std::int32_t Iter() const { return this->count_ == 0 ? 0 : this->count_ - 1; }
  DataIterProxy& operator++() {
    CHECK(this->Next());
    return *this;
  }
};

/**
 * @brief A proxy of DMatrix used by the external iterator.
 */
class DMatrixProxy : public DMatrix {
  MetaInfo info_;
  std::any batch_;
  Context ctx_;

 public:
  DeviceOrd Device() const { return ctx_.Device(); }

  /**
   * Device setters
   */
  void SetCudaColumnar(StringView data);
  void SetCudaArray(StringView data);
  /**
   * Host setters
   */
  void SetColumnar(StringView data);
  void SetArray(StringView data);
  void SetCsr(char const* c_indptr, char const* c_indices, char const* c_values,
              bst_feature_t n_features, bool on_host);

  MetaInfo& Info() override { return info_; }
  MetaInfo const& Info() const override { return info_; }
  Context const* Ctx() const override { return &ctx_; }

  [[nodiscard]] bool EllpackExists() const override { return false; }
  [[nodiscard]] bool GHistIndexExists() const override { return false; }
  [[nodiscard]] bool SparsePageExists() const override { return false; }

  template <typename Page>
  static BatchSet<Page> NoBatch() {
    LOG(FATAL) << "Proxy DMatrix cannot return data batch.";
    return BatchSet<Page>(BatchIterator<Page>(nullptr));
  }

  DMatrix* Slice(common::Span<int32_t const> /*ridxs*/) override {
    LOG(FATAL) << "Slicing DMatrix is not supported for Proxy DMatrix.";
    return nullptr;
  }
  DMatrix* SliceCol(int, int) override {
    LOG(FATAL) << "Slicing DMatrix columns is not supported for Proxy DMatrix.";
    return nullptr;
  }
  BatchSet<SparsePage> GetRowBatches() override { return NoBatch<SparsePage>(); }
  BatchSet<CSCPage> GetColumnBatches(Context const*) override { return NoBatch<CSCPage>(); }
  BatchSet<SortedCSCPage> GetSortedColumnBatches(Context const*) override {
    return NoBatch<SortedCSCPage>();
  }
  BatchSet<EllpackPage> GetEllpackBatches(Context const*, BatchParam const&) override {
    return NoBatch<EllpackPage>();
  }
  BatchSet<GHistIndexMatrix> GetGradientIndex(Context const*, BatchParam const&) override {
    return NoBatch<GHistIndexMatrix>();
  }
  BatchSet<ExtSparsePage> GetExtBatches(Context const*, BatchParam const&) override {
    return NoBatch<ExtSparsePage>();
  }
  std::any Adapter() const { return batch_; }
};

inline DMatrixProxy* MakeProxy(DMatrixHandle proxy) {
  auto proxy_handle = static_cast<std::shared_ptr<DMatrix>*>(proxy);
  CHECK(proxy_handle) << "Invalid proxy handle.";
  DMatrixProxy* typed = static_cast<DMatrixProxy*>(proxy_handle->get());
  CHECK(typed) << "Invalid proxy handle.";
  return typed;
}

/**
 * @brief Shape and basic information for data fetched from an external data iterator.
 */
struct ExternalDataInfo {
  bst_idx_t n_features = 0;               // The number of columns
  bst_idx_t n_batches = 0;                // The number of batches from the external data iterator
  bst_idx_t accumulated_rows = 0;         // The total number of rows
  bst_idx_t nnz = 0;                      // The number of non-missing values
  std::vector<bst_idx_t> column_sizes;    // The nnz for each column
  std::vector<bst_idx_t> batch_nnz;       // nnz for each batch
  std::vector<bst_idx_t> base_rowids{0};  // base_rowid
  bst_idx_t row_stride{0};                // Used by ellpack, maximum row stride for all batches
  std::shared_ptr<CatContainer> cats;     // Categories from one of the batches

  void Validate() const {
    CHECK(std::none_of(this->column_sizes.cbegin(), this->column_sizes.cend(), [&](auto f) {
      return f > this->accumulated_rows;
    })) << "Something went wrong during iteration.";

    CHECK_GE(this->n_features, 1) << "Data must has at least 1 column.";
    CHECK_EQ(this->base_rowids.size(), this->n_batches + 1);
    CHECK_LE(this->row_stride, this->n_features);
  }

  void SetInfo(Context const* ctx, bool sync, MetaInfo* p_info) {
    // From here on Info() has the correct data shape
    auto& info = *p_info;
    info.num_row_ = this->accumulated_rows;
    info.num_col_ = this->n_features;
    info.num_nonzero_ = this->nnz;
    if (sync) {
      info.SynchronizeNumberOfColumns(ctx, DataSplitMode::kRow);
    }
    info.Cats(this->cats);
    this->Validate();
  }
};

namespace cpu_impl {
/**
 * @brief Dispatch function call based on the input type.
 *
 * @tparam get_value Whether the funciton Fn accepts an adapter batch or the adapter itself.
 * @tparam AddPtrT   The type of the adapter pointer. Use std::add_pointer_t for raw pointer.
 * @tparam Fn        The type of the function to be dispatched.
 *
 * @param x     Any any object that contains a (shared) pointer to an adapter.
 * @param fn    The function to be dispatched.
 * @param type_error[out] Set to ture if it's not null and the input data is not recognized by
 *                        the host.
 *
 * @return The return value of the function being dispatched.
 */
template <bool get_value = true, template <typename A> typename AddPtrT = std::shared_ptr,
          typename Fn>
decltype(auto) DispatchAny(Context const* ctx, std::any x, Fn&& fn, bool* type_error = nullptr) {
  // CSC, FileAdapter, and IteratorAdapter are not supported.
  auto has_type = [&] {
    if (type_error) {
      *type_error = false;
    }
  };
  CHECK(x.has_value());
  if (x.type() == typeid(AddPtrT<data::DenseAdapter>)) {
    has_type();
    if constexpr (get_value) {
      auto value = std::any_cast<AddPtrT<DenseAdapter>>(x)->Value();
      return fn(value);
    } else {
      auto value = std::any_cast<AddPtrT<DenseAdapter>>(x);
      fn(value);
    }
  } else if (x.type() == typeid(AddPtrT<ArrayAdapter>)) {
    has_type();
    if constexpr (get_value) {
      auto value = std::any_cast<AddPtrT<ArrayAdapter>>(x)->Value();
      return fn(value);
    } else {
      auto value = std::any_cast<AddPtrT<ArrayAdapter>>(x);
      return fn(value);
    }
  } else if (x.type() == typeid(AddPtrT<CSRArrayAdapter>)) {
    has_type();
    if constexpr (get_value) {
      auto value = std::any_cast<AddPtrT<CSRArrayAdapter>>(x)->Value();
      return fn(value);
    } else {
      auto value = std::any_cast<AddPtrT<CSRArrayAdapter>>(x);
      return fn(value);
    }
  } else if (x.type() == typeid(AddPtrT<ColumnarAdapter>)) {
    has_type();
    auto adapter = std::any_cast<AddPtrT<ColumnarAdapter>>(x);
    if constexpr (get_value) {
      auto value = adapter->Value();
      if (adapter->HasRefCategorical()) {
        auto [batch, mapping] = MakeEncColumnarBatch(ctx, adapter);
        return fn(batch);
      }
      return fn(value);
    } else {
      return fn(adapter);
    }
  } else {
    if (type_error) {
      *type_error = true;
    } else {
      LOG(FATAL) << "Unknown type: " << x.type().name();
    }
  }

  if constexpr (get_value) {
    return std::invoke_result_t<Fn, decltype(std::declval<AddPtrT<ArrayAdapter>>()->Value())>();
  } else {
    return std::invoke_result_t<Fn, decltype(std::declval<AddPtrT<ArrayAdapter>>())>();
  }
}

template <bool get_value = true, typename Fn>
decltype(auto) DispatchAny(DMatrixProxy const* proxy, Fn&& fn, bool* type_error = nullptr) {
  return DispatchAny<get_value>(proxy->Ctx(), proxy->Adapter(), std::forward<Fn>(fn), type_error);
}

/**
 * @brief Get categories for the current batch.
 *
 * @return A host view to the categories
 */
[[nodiscard]] inline decltype(auto) BatchCats(DMatrixProxy const* proxy) {
  return DispatchAny<false>(proxy, [](auto const& adapter) -> decltype(auto) {
    using AdapterT = typename std::remove_reference_t<decltype(adapter)>::element_type;
    if constexpr (std::is_same_v<AdapterT, ColumnarAdapter>) {
      if (adapter->HasRefCategorical()) {
        return adapter->RefCats();
      }
      return adapter->Cats();
    }
    return enc::HostColumnsView{};
  });
}
}  // namespace cpu_impl

/**
 * @brief Create a `SimpleDMatrix` instance from a `DMatrixProxy`.
 *
 *    This is used for enabling inplace-predict fallback.
 */
std::shared_ptr<DMatrix> CreateDMatrixFromProxy(Context const* ctx,
                                                std::shared_ptr<DMatrixProxy> proxy, float missing);

namespace cuda_impl {
[[nodiscard]] bst_idx_t BatchSamples(DMatrixProxy const*);
[[nodiscard]] bst_idx_t BatchColumns(DMatrixProxy const*);
#if defined(XGBOOST_USE_CUDA)
[[nodiscard]] bool BatchCatsIsRef(DMatrixProxy const*);
[[nodiscard]] enc::DeviceColumnsView BatchCats(DMatrixProxy const*);
#endif  // defined(XGBOOST_USE_CUDA)
}  // namespace cuda_impl

/**
 * @brief Get the number of samples for the current batch.
 */
[[nodiscard]] inline bst_idx_t BatchSamples(DMatrixProxy const* proxy) {
  bool type_error = false;
  auto n_samples =
      cpu_impl::DispatchAny(proxy, [](auto const& value) { return value.NumRows(); }, &type_error);
  if (type_error) {
    n_samples = cuda_impl::BatchSamples(proxy);
  }
  return n_samples;
}

/**
 * @brief Get the number of features for the current batch.
 */
[[nodiscard]] inline bst_feature_t BatchColumns(DMatrixProxy const* proxy) {
  bool type_error = false;
  auto n_features =
      cpu_impl::DispatchAny(proxy, [](auto const& value) { return value.NumCols(); }, &type_error);
  if (type_error) {
    n_features = cuda_impl::BatchColumns(proxy);
  }
  return n_features;
}

namespace cpu_impl {}  // namespace cpu_impl
[[nodiscard]] bool BatchCatsIsRef(DMatrixProxy const* proxy);
}  // namespace xgboost::data
#endif  // XGBOOST_DATA_PROXY_DMATRIX_H_
