/*!
 *  Copyright (c) 2019 by Contributors
 * \file device_adapter.cuh
 */
#ifndef XGBOOST_DATA_DEVICE_ADAPTER_H_
#define XGBOOST_DATA_DEVICE_ADAPTER_H_
#include <limits>
#include <memory>
#include <string>
#include "../common/device_helpers.cuh"
#include "../common/math.h"
#include "adapter.h"
#include "array_interface.h"

namespace xgboost {
namespace data {

struct IsValidFunctor : public thrust::unary_function<Entry, bool> {
  explicit IsValidFunctor(float missing) : missing(missing) {}

  float missing;
  __device__ bool operator()(const data::COOTuple& e) const {
    if (common::CheckNAN(e.value) || e.value == missing) {
      return false;
    }
    return true;
  }
  __device__ bool operator()(const Entry& e) const {
    if (common::CheckNAN(e.fvalue) || e.fvalue == missing) {
      return false;
    }
    return true;
  }
};

class CudfAdapterBatch : public detail::NoMetaInfo {
  friend class CudfAdapter;

 public:
  CudfAdapterBatch() = default;
  CudfAdapterBatch(common::Span<ArrayInterface> columns, size_t num_rows)
      : columns_(columns),
        num_rows_(num_rows) {}
  size_t Size() const { return num_rows_ * columns_.size(); }
  __device__ COOTuple GetElement(size_t idx) const {
    size_t column_idx = idx % columns_.size();
    size_t row_idx = idx / columns_.size();
    auto const& column = columns_[column_idx];
    float value = column.valid.Data() == nullptr || column.valid.Check(row_idx)
                      ? column.GetElement(row_idx)
                      : std::numeric_limits<float>::quiet_NaN();
    return {row_idx, column_idx, value};
  }

  XGBOOST_DEVICE bst_row_t NumRows() const { return num_rows_; }
  XGBOOST_DEVICE bst_row_t NumCols() const { return columns_.size(); }

 private:
  common::Span<ArrayInterface> columns_;
  size_t num_rows_;
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
  explicit CudfAdapter(std::string cuda_interfaces_str) {
    Json interfaces =
        Json::Load({cuda_interfaces_str.c_str(), cuda_interfaces_str.size()});
    std::vector<Json> const& json_columns = get<Array>(interfaces);
    size_t n_columns = json_columns.size();
    CHECK_GT(n_columns, 0) << "Number of columns must not equal to 0.";

    auto const& typestr = get<String const>(json_columns[0]["typestr"]);
    CHECK_EQ(typestr.size(), 3) << ArrayInterfaceErrors::TypestrFormat();
    CHECK_NE(typestr.front(), '>') << ArrayInterfaceErrors::BigEndian();
    std::vector<ArrayInterface> columns;
    auto first_column = ArrayInterface(get<Object const>(json_columns[0]));
    num_rows_ = first_column.num_rows;
    if (num_rows_ == 0) {
      return;
    }

    device_idx_ = dh::CudaGetPointerDevice(first_column.data);
    CHECK_NE(device_idx_, -1);
    dh::safe_cuda(cudaSetDevice(device_idx_));
    for (auto& json_col : json_columns) {
      auto column = ArrayInterface(get<Object const>(json_col));
      columns.push_back(column);
      CHECK_EQ(column.num_cols, 1);
      num_rows_ = std::max(num_rows_, size_t(column.num_rows));
      CHECK_EQ(device_idx_, dh::CudaGetPointerDevice(column.data))
          << "All columns should use the same device.";
      CHECK_EQ(num_rows_, column.num_rows)
          << "All columns should have same number of rows.";
    }
    columns_ = columns;
    batch_ = CudfAdapterBatch(dh::ToSpan(columns_), num_rows_);
  }
  const CudfAdapterBatch& Value() const override {
    CHECK_EQ(batch_.columns_.data(), columns_.data().get());
    return batch_;
  }

  size_t NumRows() const { return num_rows_; }
  size_t NumColumns() const { return columns_.size(); }
  size_t DeviceIdx() const { return device_idx_; }

 private:
  CudfAdapterBatch batch_;
  dh::device_vector<ArrayInterface> columns_;
  size_t num_rows_{0};
  int device_idx_;
};

class CupyAdapterBatch : public detail::NoMetaInfo {
 public:
  CupyAdapterBatch() = default;
  explicit CupyAdapterBatch(ArrayInterface array_interface)
    : array_interface_(std::move(array_interface)) {}
  size_t Size() const {
    return array_interface_.num_rows * array_interface_.num_cols;
  }
  __device__ COOTuple GetElement(size_t idx) const {
    size_t column_idx = idx % array_interface_.num_cols;
    size_t row_idx = idx / array_interface_.num_cols;
    float value = array_interface_.GetElement(idx);
    return {row_idx, column_idx, value};
  }

  XGBOOST_DEVICE bst_row_t NumRows() const { return array_interface_.num_rows; }
  XGBOOST_DEVICE bst_row_t NumCols() const { return array_interface_.num_cols; }

 private:
  ArrayInterface array_interface_;
};

class CupyAdapter : public detail::SingleBatchDataIter<CupyAdapterBatch> {
 public:
  explicit CupyAdapter(std::string cuda_interface_str) {
    Json json_array_interface =
        Json::Load({cuda_interface_str.c_str(), cuda_interface_str.size()});
    array_interface_ = ArrayInterface(get<Object const>(json_array_interface), false);
    batch_ = CupyAdapterBatch(array_interface_);
    if (array_interface_.num_rows == 0) {
      return;
    }
    device_idx_ = dh::CudaGetPointerDevice(array_interface_.data);
    CHECK_NE(device_idx_, -1);
  }
  const CupyAdapterBatch& Value() const override { return batch_; }

  size_t NumRows() const { return array_interface_.num_rows; }
  size_t NumColumns() const { return array_interface_.num_cols; }
  size_t DeviceIdx() const { return device_idx_; }

 private:
  ArrayInterface array_interface_;
  CupyAdapterBatch batch_;
  int device_idx_;
};

// Returns maximum row length
template <typename AdapterBatchT>
size_t GetRowCounts(const AdapterBatchT batch, common::Span<size_t> offset,
                    int device_idx, float missing) {
  IsValidFunctor is_valid(missing);
  // Count elements per row
  dh::LaunchN(device_idx, batch.Size(), [=] __device__(size_t idx) {
    auto element = batch.GetElement(idx);
    if (is_valid(element)) {
      atomicAdd(reinterpret_cast<unsigned long long*>(  // NOLINT
                    &offset[element.row_idx]),
                static_cast<unsigned long long>(1));  // NOLINT
    }
  });
  dh::XGBCachingDeviceAllocator<char> alloc;
  size_t row_stride = dh::Reduce(
      thrust::cuda::par(alloc), thrust::device_pointer_cast(offset.data()),
      thrust::device_pointer_cast(offset.data()) + offset.size(), size_t(0),
      thrust::maximum<size_t>());
  return row_stride;
}
};  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_DEVICE_ADAPTER_H_
