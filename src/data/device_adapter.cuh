/**
 *  Copyright 2019-2023 by XGBoost Contributors
 * \file device_adapter.cuh
 */
#ifndef XGBOOST_DATA_DEVICE_ADAPTER_H_
#define XGBOOST_DATA_DEVICE_ADAPTER_H_
#include <thrust/iterator/counting_iterator.h>  // for make_counting_iterator
#include <thrust/logical.h>                     // for none_of

#include <cstddef>                              // for size_t
#include <limits>
#include <memory>
#include <string>

#include "../common/device_helpers.cuh"
#include "../common/math.h"
#include "adapter.h"
#include "array_interface.h"

namespace xgboost {
namespace data {

class CudfAdapterBatch : public detail::NoMetaInfo {
  friend class CudfAdapter;

 public:
  CudfAdapterBatch() = default;
  CudfAdapterBatch(common::Span<ArrayInterface<1>> columns, size_t num_rows)
      : columns_(columns),
        num_rows_(num_rows) {}
  size_t Size() const { return num_rows_ * columns_.size(); }
  __device__ __forceinline__ COOTuple GetElement(size_t idx) const {
    size_t column_idx = idx % columns_.size();
    size_t row_idx = idx / columns_.size();
    auto const& column = columns_[column_idx];
    float value = column.valid.Data() == nullptr || column.valid.Check(row_idx)
                      ? column(row_idx)
                      : std::numeric_limits<float>::quiet_NaN();
    return {row_idx, column_idx, value};
  }

  XGBOOST_DEVICE bst_row_t NumRows() const { return num_rows_; }
  XGBOOST_DEVICE bst_row_t NumCols() const { return columns_.size(); }

 private:
  common::Span<ArrayInterface<1>> columns_;
  size_t num_rows_{0};
};

/*!
 * Please be careful that, in official specification, the only three required
 * fields are `shape', `version' and `typestr'.  Any other is optional,
 * including `data'.  But here we have one additional requirements for input
 * data:
 *
 * - `data' field is required, passing in an empty dataset is not accepted, as
 * most (if not all) of our algorithms don't have test for empty dataset.  An
 * error is better than a crash.
 *
 * What if invalid value from dataframe is 0 but I specify missing=NaN in
 * XGBoost?  Since validity mask is ignored, all 0s are preserved in XGBoost.
 *
 * FIXME(trivialfis): Put above into document after we have a consistent way for
 * processing input data.
 *
 * Sample input:
 * [
 *   {
 *     "shape": [
 *       10
 *     ],
 *     "strides": [
 *       4
 *     ],
 *     "data": [
 *       30074864128,
 *       false
 *     ],
 *     "typestr": "<f4",
 *     "version": 1,
 *     "mask": {
 *       "shape": [
 *         64
 *       ],
 *       "strides": [
 *         1
 *       ],
 *       "data": [
 *         30074864640,
 *         false
 *       ],
 *       "typestr": "|i1",
 *       "version": 1
 *     }
 *   }
 * ]
 */
class CudfAdapter : public detail::SingleBatchDataIter<CudfAdapterBatch> {
 public:
  explicit CudfAdapter(StringView cuda_interfaces_str) {
    Json interfaces = Json::Load(cuda_interfaces_str);
    std::vector<Json> const& json_columns = get<Array>(interfaces);
    size_t n_columns = json_columns.size();
    CHECK_GT(n_columns, 0) << "Number of columns must not equal to 0.";

    auto const& typestr = get<String const>(json_columns[0]["typestr"]);
    CHECK_EQ(typestr.size(), 3) << ArrayInterfaceErrors::TypestrFormat();
    std::vector<ArrayInterface<1>> columns;
    auto first_column = ArrayInterface<1>(get<Object const>(json_columns[0]));
    num_rows_ = first_column.Shape(0);
    if (num_rows_ == 0) {
      return;
    }

    device_idx_ = dh::CudaGetPointerDevice(first_column.data);
    CHECK_NE(device_idx_, Context::kCpuId);
    dh::safe_cuda(cudaSetDevice(device_idx_));
    for (auto& json_col : json_columns) {
      auto column = ArrayInterface<1>(get<Object const>(json_col));
      columns.push_back(column);
      num_rows_ = std::max(num_rows_, column.Shape(0));
      CHECK_EQ(device_idx_, dh::CudaGetPointerDevice(column.data))
          << "All columns should use the same device.";
      CHECK_EQ(num_rows_, column.Shape(0))
          << "All columns should have same number of rows.";
    }
    columns_ = columns;
    batch_ = CudfAdapterBatch(dh::ToSpan(columns_), num_rows_);
  }
  explicit CudfAdapter(std::string cuda_interfaces_str)
      : CudfAdapter{StringView{cuda_interfaces_str}} {}

  const CudfAdapterBatch& Value() const override {
    CHECK_EQ(batch_.columns_.data(), columns_.data().get());
    return batch_;
  }

  size_t NumRows() const { return num_rows_; }
  size_t NumColumns() const { return columns_.size(); }
  int32_t DeviceIdx() const { return device_idx_; }

 private:
  CudfAdapterBatch batch_;
  dh::device_vector<ArrayInterface<1>> columns_;
  size_t num_rows_{0};
  int32_t device_idx_{Context::kCpuId};
};

class CupyAdapterBatch : public detail::NoMetaInfo {
 public:
  CupyAdapterBatch() = default;
  explicit CupyAdapterBatch(ArrayInterface<2> array_interface)
    : array_interface_(std::move(array_interface)) {}
  size_t Size() const {
    return array_interface_.Shape(0) * array_interface_.Shape(1);
  }
  __device__ COOTuple GetElement(size_t idx) const {
    size_t column_idx = idx % array_interface_.Shape(1);
    size_t row_idx = idx / array_interface_.Shape(1);
    float value = array_interface_(row_idx, column_idx);
    return {row_idx, column_idx, value};
  }

  XGBOOST_DEVICE bst_row_t NumRows() const { return array_interface_.Shape(0); }
  XGBOOST_DEVICE bst_row_t NumCols() const { return array_interface_.Shape(1); }

 private:
  ArrayInterface<2> array_interface_;
};

class CupyAdapter : public detail::SingleBatchDataIter<CupyAdapterBatch> {
 public:
  explicit CupyAdapter(StringView cuda_interface_str) {
    Json json_array_interface = Json::Load(cuda_interface_str);
    array_interface_ = ArrayInterface<2>(get<Object const>(json_array_interface));
    batch_ = CupyAdapterBatch(array_interface_);
    if (array_interface_.Shape(0) == 0) {
      return;
    }
    device_idx_ = dh::CudaGetPointerDevice(array_interface_.data);
    CHECK_NE(device_idx_, Context::kCpuId);
  }
  explicit CupyAdapter(std::string cuda_interface_str)
      : CupyAdapter{StringView{cuda_interface_str}} {}
  const CupyAdapterBatch& Value() const override { return batch_; }

  size_t NumRows() const { return array_interface_.Shape(0); }
  size_t NumColumns() const { return array_interface_.Shape(1); }
  int32_t DeviceIdx() const { return device_idx_; }

 private:
  ArrayInterface<2> array_interface_;
  CupyAdapterBatch batch_;
  int32_t device_idx_ {Context::kCpuId};
};

// Returns maximum row length
template <typename AdapterBatchT>
size_t GetRowCounts(const AdapterBatchT batch, common::Span<size_t> offset,
                    int device_idx, float missing) {
  dh::safe_cuda(cudaSetDevice(device_idx));
  IsValidFunctor is_valid(missing);
  // Count elements per row
  dh::LaunchN(batch.Size(), [=] __device__(size_t idx) {
    auto element = batch.GetElement(idx);
    if (is_valid(element)) {
      atomicAdd(reinterpret_cast<unsigned long long*>(  // NOLINT
                    &offset[element.row_idx]),
                static_cast<unsigned long long>(1));  // NOLINT
    }
  });
  dh::XGBCachingDeviceAllocator<char> alloc;
  size_t row_stride =
      dh::Reduce(thrust::cuda::par(alloc), thrust::device_pointer_cast(offset.data()),
                 thrust::device_pointer_cast(offset.data()) + offset.size(),
                 static_cast<std::size_t>(0), thrust::maximum<size_t>());
  return row_stride;
}

/**
 * \brief Check there's no inf in data.
 */
template <typename AdapterBatchT>
bool NoInfInData(AdapterBatchT const& batch, IsValidFunctor is_valid) {
  auto counting = thrust::make_counting_iterator(0llu);
  auto value_iter = dh::MakeTransformIterator<bool>(counting, [=] XGBOOST_DEVICE(std::size_t idx) {
    auto v = batch.GetElement(idx).value;
    if (!is_valid(v)) {
      // discard the invalid elements.
      return true;
    }
    // check that there's no inf in data.
    return !std::isinf(v);
  });
  dh::XGBCachingDeviceAllocator<char> alloc;
  // The default implementation in thrust optimizes any_of/none_of/all_of by using small
  // intervals to early stop. But we expect all data to be valid here, using small
  // intervals only decreases performance due to excessive kernel launch and stream
  // synchronization.
  auto valid = dh::Reduce(thrust::cuda::par(alloc), value_iter, value_iter + batch.Size(), true,
                          thrust::logical_and<>{});
  return valid;
}
};  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_DEVICE_ADAPTER_H_
