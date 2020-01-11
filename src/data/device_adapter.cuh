/*!
 *  Copyright (c) 2019 by Contributors
 * \file device_adapter.cuh
 */
#ifndef XGBOOST_DATA_DEVICE_ADAPTER_H_
#define XGBOOST_DATA_DEVICE_ADAPTER_H_
#include <limits>
#include <memory>
#include <string>
#include "columnar.h"
#include "adapter.h"
#include "../common/device_helpers.cuh"

namespace xgboost {
namespace data {

class CudfAdapterBatch : public detail::NoMetaInfo {
 public:
  CudfAdapterBatch() = default;
  CudfAdapterBatch(common::Span<Columnar> columns,
                   common::Span<size_t> column_ptr, size_t num_elements)
      : columns_(columns),column_ptr_(column_ptr), num_elements(num_elements) {}
  size_t Size()const  { return num_elements; }
  __device__ COOTuple GetElement(size_t idx)const
  {
    size_t column_idx =
        dh::UpperBound(column_ptr_.data(), column_ptr_.size(), idx) - 1;
    auto& column = columns_[column_idx];
    size_t row_idx = idx - column_ptr_[column_idx];
    float value = column.valid.Data() == nullptr || column.valid.Check(row_idx)
                      ? column.GetElement(row_idx)
                      : std::numeric_limits<float>::quiet_NaN();
    return COOTuple(row_idx, column_idx, value);
  }

 private:
  common::Span<Columnar> columns_;
  common::Span<size_t> column_ptr_;
  size_t num_elements;
};

/*!
 * Please be careful that, in official specification, the only three required fields are
 * `shape', `version' and `typestr'.  Any other is optional, including `data'.  But here
 * we have one additional requirements for input data:
 *
 * - `data' field is required, passing in an empty dataset is not accepted, as most (if
 *   not all) of our algorithms don't have test for empty dataset.  An error is better
 *   than a crash.
 *
 * What if invalid value from dataframe is 0 but I specify missing=NaN in XGBoost?  Since
 * validity mask is ignored, all 0s are preserved in XGBoost.
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
    CHECK_EQ(typestr.size(), 3) << ColumnarErrors::TypestrFormat();
    CHECK_NE(typestr.front(), '>') << ColumnarErrors::BigEndian();
    std::vector<Columnar> columns;
    std::vector<size_t> column_ptr({0});
    auto first_column = Columnar(get<Object const>(json_columns[0]));
    device_idx_ = dh::CudaGetPointerDevice(first_column.data);
    CHECK_NE(device_idx_, -1);
    dh::safe_cuda(cudaSetDevice(device_idx_));
    num_rows_ = first_column.size;
    for (auto& json_col : json_columns) {
      auto column = Columnar(get<Object const>(json_col));
      columns.push_back(column);
      column_ptr.emplace_back(column_ptr.back() + column.size);
      num_rows_ = std::max(num_rows_, size_t(column.size));
      CHECK_EQ(device_idx_, dh::CudaGetPointerDevice(column.data))
          << "All columns should use the same device.";
      CHECK_EQ(num_rows_, column.size)
          << "All columns should have same number of rows.";
    }
    columns_ = columns;
    column_ptr_ = column_ptr;
    batch = CudfAdapterBatch(dh::ToSpan(columns_), dh::ToSpan(column_ptr_),
                             column_ptr.back());
  }
  const CudfAdapterBatch& Value() const override { return batch; }

  size_t NumRows() const { return num_rows_; }
  size_t NumColumns() const { return columns_.size(); }
  size_t DeviceIdx()const {
    return device_idx_;
  }

  // Cudf is column major
  bool IsRowMajor() { return false; }
 private:
  CudfAdapterBatch batch;
  dh::device_vector<Columnar> columns_;
  dh::device_vector<size_t> column_ptr_;  // Exclusive scan of column sizes
  size_t num_rows_{0};
  int device_idx_;
};
};  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_DEVICE_ADAPTER_H_
