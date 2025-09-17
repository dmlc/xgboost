/**
 * Copyright 2019-2025, XGBoost Contributors
 * @file device_adapter.cuh
 */
#ifndef XGBOOST_DATA_DEVICE_ADAPTER_H_
#define XGBOOST_DATA_DEVICE_ADAPTER_H_

#include <thrust/functional.h>                   // for maximum

#include <cstddef>           // for size_t
#include <cuda/std/variant>  // for variant
#include <limits>            // for numeric_limits
#include <memory>            // for make_unique
#include <string>            // for string

#include "../common/algorithm.cuh"  // for AllOf
#include "../common/cuda_context.cuh"
#include "../common/device_helpers.cuh"
#include "adapter.h"
#include "array_interface.h"
#include "cat_container.cuh"      // for MakeCatAccessor
#include "xgboost/string_view.h"  // for StringView

namespace xgboost::data {
template <typename EncAccessor>
class EncCudfAdapterBatchImpl : public detail::NoMetaInfo {
 private:
  common::Span<ArrayInterface<1> const> columns_;
  bst_idx_t n_samples_{0};
  EncAccessor acc_;

 public:
  EncCudfAdapterBatchImpl() = default;
  EncCudfAdapterBatchImpl(common::Span<ArrayInterface<1> const> columns, EncAccessor acc,
                          bst_idx_t n_samples)
      : columns_(columns), n_samples_(n_samples), acc_{std::move(acc)} {}
  [[nodiscard]] std::size_t Size() const { return n_samples_ * columns_.size(); }
  [[nodiscard]] __device__ __forceinline__ COOTuple GetElement(bst_idx_t idx) const {
    auto column_idx = idx % columns_.size();
    auto row_idx = idx / columns_.size();
    auto value = this->GetElement(row_idx, column_idx);
    return {row_idx, column_idx, value};
  }

  [[nodiscard]] __device__ float GetElement(bst_idx_t ridx, bst_feature_t fidx) const {
    auto const& column = columns_[fidx];
    float value = column.valid.Data() == nullptr || column.valid.Check(ridx)
                      ? column(ridx)
                      : std::numeric_limits<float>::quiet_NaN();
    return acc_(value, fidx);
  }

  [[nodiscard]] XGBOOST_DEVICE bst_idx_t NumRows() const { return n_samples_; }
  [[nodiscard]] XGBOOST_DEVICE bst_idx_t NumCols() const { return columns_.size(); }
  [[nodiscard]] common::Span<ArrayInterface<1> const> Columns() const { return this->columns_; }
};

using CudfAdapterBatch = EncCudfAdapterBatchImpl<NoOpAccessor>;
using EncCudfAdapterBatch = EncCudfAdapterBatchImpl<CatAccessor>;

/**
 * @brief Device columnar format. We call it cuDF, but it's just arrow-CUDA since cuDF
 * adopts the arrow format.
 *
 * See @ref XGDMatrixCreateFromColumnar for notes
 */
class CudfAdapter : public detail::SingleBatchDataIter<CudfAdapterBatch> {
 public:
  explicit CudfAdapter(StringView cuda_interfaces_str);
  explicit CudfAdapter(std::string cuda_interfaces_str)
      : CudfAdapter{StringView{cuda_interfaces_str}} {}

  [[nodiscard]] CudfAdapterBatch const& Value() const override {
    CHECK_EQ(batch_.Columns().data(), columns_.data().get());
    return batch_;
  }

  [[nodiscard]] std::size_t NumRows() const { return num_rows_; }
  [[nodiscard]] std::size_t NumColumns() const { return columns_.size(); }
  [[nodiscard]] DeviceOrd Device() const { return device_; }
  [[nodiscard]] bst_idx_t SizeBytes() const { return this->n_bytes_; }

  [[nodiscard]] enc::DeviceColumnsView Cats() const {
    return {common::Span{this->cats_}, dh::ToSpan(this->cat_segments_), this->n_total_cats_};
  }
  [[nodiscard]] enc::DeviceColumnsView DCats() const {
    return {dh::ToSpan(this->d_cats_), dh::ToSpan(this->cat_segments_), this->n_total_cats_};
  }
  [[nodiscard]] enc::DeviceColumnsView RefCats() const { return ref_cats_; }
  [[nodiscard]] bool HasCategorical() const { return n_total_cats_ != 0; }
  [[nodiscard]] bool HasRefCategorical() const { return this->ref_cats_.n_total_cats != 0; }

  [[nodiscard]] common::Span<ArrayInterface<1> const> Columns() const {
    return dh::ToSpan(this->columns_);
  }

 private:
  CudfAdapterBatch batch_;
  dh::device_vector<ArrayInterface<1>> columns_;

  // Categories
  std::vector<enc::DeviceCatIndexView> cats_;
  dh::device_vector<enc::DeviceCatIndexView> d_cats_;
  dh::device_vector<std::int32_t> cat_segments_;
  std::int32_t n_total_cats_{0};

  enc::DeviceColumnsView ref_cats_;                  // A view to the reference category.
  std::vector<enc::DeviceCatIndexView> h_ref_cats_;  // host storage for column view

  size_t num_rows_{0};
  bst_idx_t n_bytes_{0};
  DeviceOrd device_{DeviceOrd::CPU()};
};

class CupyAdapterBatch : public detail::NoMetaInfo {
 public:
  CupyAdapterBatch() = default;
  explicit CupyAdapterBatch(ArrayInterface<2> array_interface)
      : array_interface_(std::move(array_interface)) {}
  // The total number of elements.
  [[nodiscard]] std::size_t Size() const {
    return array_interface_.Shape<0>() * array_interface_.Shape<1>();
  }
  [[nodiscard]] __device__ COOTuple GetElement(size_t idx) const {
    size_t column_idx = idx % array_interface_.Shape<1>();
    size_t row_idx = idx / array_interface_.Shape<1>();
    float value = array_interface_(row_idx, column_idx);
    return {row_idx, column_idx, value};
  }
  [[nodiscard]] __device__ float GetElement(bst_idx_t ridx, bst_feature_t fidx) const {
    float value = array_interface_(ridx, fidx);
    return value;
  }

  [[nodiscard]] XGBOOST_DEVICE bst_idx_t NumRows() const { return array_interface_.Shape<0>(); }
  [[nodiscard]] XGBOOST_DEVICE bst_idx_t NumCols() const { return array_interface_.Shape<1>(); }

 private:
  ArrayInterface<2> array_interface_;
};

inline auto MakeEncColumnarBatch(Context const* ctx, CudfAdapter const* adapter) {
  auto cats = std::make_unique<CatContainer>(ctx, adapter->RefCats(), true);
  cats->Sort(ctx);
  auto [acc, mapping] = ::xgboost::cuda_impl::MakeCatAccessor(ctx, adapter->DCats(), cats.get());
  return std::tuple{EncCudfAdapterBatch{adapter->Columns(), acc, adapter->NumRows()},
                    std::move(mapping)};
}

inline auto MakeEncColumnarBatch(Context const* ctx, std::shared_ptr<CudfAdapter> const& adapter) {
  return MakeEncColumnarBatch(ctx, adapter.get());
}

class CupyAdapter : public detail::SingleBatchDataIter<CupyAdapterBatch> {
 public:
  explicit CupyAdapter(StringView cuda_interface_str) {
    Json json_array_interface = Json::Load(cuda_interface_str);
    array_interface_ = ArrayInterface<2>(get<Object const>(json_array_interface));
    batch_ = CupyAdapterBatch(array_interface_);
    if (array_interface_.Shape<0>() == 0) {
      return;
    }
    device_ = DeviceOrd::CUDA(dh::CudaGetPointerDevice(array_interface_.data));
    this->n_bytes_ =
        array_interface_.Shape<0>() * array_interface_.Shape<1>() * array_interface_.ElementSize();
    CHECK(device_.IsCUDA());
  }
  explicit CupyAdapter(std::string cuda_interface_str)
      : CupyAdapter{StringView{cuda_interface_str}} {}
  [[nodiscard]] const CupyAdapterBatch& Value() const override { return batch_; }

  [[nodiscard]] std::size_t NumRows() const { return array_interface_.Shape<0>(); }
  [[nodiscard]] std::size_t NumColumns() const { return array_interface_.Shape<1>(); }
  [[nodiscard]] DeviceOrd Device() const { return device_; }
  [[nodiscard]] bst_idx_t SizeBytes() const { return this->n_bytes_; }

 private:
  ArrayInterface<2> array_interface_;
  CupyAdapterBatch batch_;
  bst_idx_t n_bytes_{0};
  DeviceOrd device_{DeviceOrd::CPU()};
};

// Returns maximum row length
template <typename AdapterBatchT>
bst_idx_t GetRowCounts(Context const* ctx, const AdapterBatchT batch,
                       common::Span<bst_idx_t> offset, DeviceOrd device, float missing) {
  dh::safe_cuda(cudaSetDevice(device.ordinal));
  IsValidFunctor is_valid(missing);
  dh::safe_cuda(
      cudaMemsetAsync(offset.data(), '\0', offset.size_bytes(), ctx->CUDACtx()->Stream()));

  auto n_samples = batch.NumRows();
  bst_feature_t n_features = batch.NumCols();

  // Use more than 1 threads for each row in case of dataset being too wide.
  bst_feature_t stride{0};
  if (n_features < 32) {
    stride = std::min(n_features, 4u);
  } else if (n_features < 64) {
    stride = 8;
  } else if (n_features < 128) {
    stride = 16;
  } else {
    stride = 32;
  }

  // Count elements per row
  dh::LaunchN(n_samples * stride, ctx->CUDACtx()->Stream(), [=] __device__(std::size_t idx) {
    bst_idx_t cnt{0};
    auto [ridx, fbeg] = linalg::UnravelIndex(idx, n_samples, stride);
    SPAN_CHECK(ridx < n_samples);
    for (bst_feature_t fidx = fbeg; fidx < n_features; fidx += stride) {
      if (is_valid(batch.GetElement(ridx, fidx))) {
        cnt++;
      }
    }

    atomicAdd(reinterpret_cast<unsigned long long*>(  // NOLINT
                  &offset[ridx]),
              static_cast<unsigned long long>(cnt));  // NOLINT
  });
  bst_idx_t row_stride =
      dh::Reduce(ctx->CUDACtx()->CTP(), thrust::device_pointer_cast(offset.data()),
                 thrust::device_pointer_cast(offset.data()) + offset.size(),
                 static_cast<bst_idx_t>(0), thrust::maximum<bst_idx_t>());
  return row_stride;
}

/**
 * @brief Check there's no inf in data.
 */
template <typename AdapterBatchT>
bool NoInfInData(Context const* ctx, AdapterBatchT const& batch, IsValidFunctor is_valid) {
  auto it = dh::MakeIndexTransformIter(
      [=] XGBOOST_DEVICE(std::size_t idx) { return batch.GetElement(idx).value; });
  return common::AllOf(ctx->CUDACtx()->CTP(), it, it + batch.Size(), [=] XGBOOST_DEVICE(float v) {
    if (is_valid(v) && isinf(v)) {
      return false;
    }
    return true;
  });
}
}  // namespace xgboost::data
#endif  // XGBOOST_DATA_DEVICE_ADAPTER_H_
