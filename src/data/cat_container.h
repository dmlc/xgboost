/**
 * Copyright 2025, XGBoost Contributors
 */
#pragma once

#include <xgboost/base.h>  // for bst_cat_t

#include <cstdint>  // for int32_t, int8_t
#include <memory>   // for unique_ptr
#include <mutex>    // for mutex
#include <string>   // for string
#include <tuple>    // for tuple
#include <vector>   // for vector

#include "../encoder/ordinal.h"          // for CatStrArrayView
#include "../encoder/types.h"            // for Overloaded
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/json.h"                // for Json

namespace xgboost {
/**
 * @brief Error policy class used to interface with the encoder implementaion.
 */
struct EncErrorPolicy {
  void Error(std::string&& msg) const { LOG(FATAL) << msg; }
};

namespace cpu_impl {
struct CatStrArray {
  std::vector<std::int32_t> offsets;
  std::vector<enc::CatCharT> values;

  [[nodiscard]] explicit operator enc::CatStrArrayView() const { return {offsets, values}; }
  [[nodiscard]] std::size_t size() const {  // NOLINT
    return enc::CatStrArrayView(*this).size();
  }
};

// Type mapping from the CPU view type to the storage type.
template <typename T>
struct ViewToStorageImpl;

template <>
struct ViewToStorageImpl<enc::CatStrArrayView> {
  using Type = CatStrArray;
};

template <typename T>
struct ViewToStorageImpl<common::Span<T const>> {
  using Type = std::vector<T>;
};

template <typename... Ts>
struct ViewToStorage;

template <typename... Ts>
struct ViewToStorage<std::tuple<Ts...>> {
  using Type = std::tuple<typename ViewToStorageImpl<Ts>::Type...>;
};

// storage type list (tuple), used for meta programming.
using CatIndexTypes = ViewToStorage<enc::CatIndexViewTypes>::Type;
// std::variant of the storage types, used for actual storage.
using ColumnType = enc::cpu_impl::TupToVarT<CatIndexTypes>;

/**
 * @brief CPU storage for categories.
 */
struct CatContainerImpl {
  std::vector<ColumnType> columns;
  // View
  std::vector<enc::HostCatIndexView> columns_v;

  void Finalize() {
    this->columns_v.clear();
    for (auto const& col : this->columns) {
      std::visit(enc::Overloaded{[this](CatStrArray const& str) {
                                   this->columns_v.emplace_back(enc::CatStrArrayView(str));
                                 },
                                 [this](auto&& values) {
                                   this->columns_v.emplace_back(common::Span{values});
                                 }},
                 col);
    }
  }

  void Copy(CatContainerImpl const* that) {
    this->columns = that->columns;
    this->Finalize();
  }
};

using EncPolicyT = enc::Policy<EncErrorPolicy>;

inline EncPolicyT EncPolicy = EncPolicyT{};
};  // namespace cpu_impl

namespace cuda_impl {
struct CatContainerImpl;
}

/**
 * @brief A container class for user-provided categories (usually from a DataFrame).
 */
class CatContainer {
  /**
   * @brief Implementation of the Copy method, used by both CPU and GPU. Note that this
   * method changes the permission in the HostDeviceVector as we need to pull data into
   * targeted devices.
   */
  void CopyCommon(Context const* ctx, CatContainer const& that) {
    auto device = ctx->Device();

    that.sorted_idx_.SetDevice(device);
    this->sorted_idx_.SetDevice(device);
    this->sorted_idx_.Resize(that.sorted_idx_.Size());
    this->sorted_idx_.Copy(that.sorted_idx_);

    this->feature_segments_.SetDevice(device);
    that.feature_segments_.SetDevice(device);
    this->feature_segments_.Resize(that.feature_segments_.Size());
    this->feature_segments_.Copy(that.feature_segments_);

    this->n_total_cats_ = that.n_total_cats_;

    if (!device.IsCPU()) {
      // Pull to device
      this->sorted_idx_.ConstDevicePointer();
      this->feature_segments_.ConstDevicePointer();
    }
  }

  [[nodiscard]] enc::HostColumnsView HostViewImpl() const {
    CHECK_EQ(this->cpu_impl_->columns.size(), this->cpu_impl_->columns_v.size());
    if (this->n_total_cats_ != 0) {
      CHECK(!this->cpu_impl_->columns_v.empty());
    }
    return {common::Span{this->cpu_impl_->columns_v}, this->feature_segments_.ConstHostSpan(),
            this->n_total_cats_};
  }

 public:
  CatContainer();
  explicit CatContainer(enc::HostColumnsView const& df);
#if defined(XGBOOST_USE_CUDA)
  explicit CatContainer(DeviceOrd device, enc::DeviceColumnsView const& df);
#endif  // defined(XGBOOST_USE_CUDA)
  ~CatContainer();

  void Copy(Context const* ctx, CatContainer const& that);

  [[nodiscard]] bool HostCanRead() const { return this->feature_segments_.HostCanRead(); }
  [[nodiscard]] bool DeviceCanRead() const { return this->feature_segments_.DeviceCanRead(); }

  // Mostly used for testing.
  void Push(cpu_impl::ColumnType const& column) { this->cpu_impl_->columns.emplace_back(column); }
  /**
   * @brief Wether the container is initialized at all. If the input is not a DataFrame,
   *        this method returns True.
   */
  [[nodiscard]] bool Empty() const;

  [[nodiscard]] std::size_t NumFeatures() const { return this->cpu_impl_->columns.size(); }
  /**
   * @brief The number of categories across all features.
   */
  [[nodiscard]] std::size_t NumCatsTotal() const { return this->n_total_cats_; }

  /**
   * @brief Sort the categories using argsort.
   *
   * This provides a common ordering of the categories between the training dataset and
   * the test dataset.
   */
  void Sort(Context const* ctx);
  /**
   * @brief Obtain a view to the sorted index created by the @ref Sort method.
   */
  [[nodiscard]] common::Span<bst_cat_t const> RefSortedIndex(Context const* ctx) const {
    std::lock_guard guard{device_mu_};
    if (ctx->IsCPU()) {
      return this->sorted_idx_.ConstHostSpan();
    } else {
      sorted_idx_.SetDevice(ctx->Device());
      return this->sorted_idx_.ConstDeviceSpan();
    }
  }
  /**
   * @brief Whether there's a categorical feature. If not,then all columns in this
   * container is empty.
   */
  [[nodiscard]] bool HasCategorical() const { return this->n_total_cats_ != 0; }

  // IO
  void Save(Json* out) const;
  void Load(Json const& in);
  /**
   * @brief Get a view to the CPU storage.
   */
  [[nodiscard]] enc::HostColumnsView HostView() const;

#if defined(XGBOOST_USE_CUDA)
  /**
   * @brief Get a view to the GPU storage.
   */
  [[nodiscard]] enc::DeviceColumnsView DeviceView(Context const* ctx) const;
#endif  // defined(XGBOOST_USE_CUDA)

 private:
  mutable std::mutex device_mu_;  // mutex for copying between devices.
  HostDeviceVector<std::int32_t> feature_segments_;
  bst_cat_t n_total_cats_{0};

  std::unique_ptr<cpu_impl::CatContainerImpl> cpu_impl_;

  HostDeviceVector<bst_cat_t> sorted_idx_;
#if defined(XGBOOST_USE_CUDA)
  std::unique_ptr<cuda_impl::CatContainerImpl> cu_impl_;
#endif  // defined(XGBOOST_USE_CUDA)
};
}  // namespace xgboost
