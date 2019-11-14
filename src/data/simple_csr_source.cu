/*!
 * Copyright 2019 by XGBoost Contributors
 *
 * \file simple_csr_source.cuh
 * \brief An extension for the simple CSR source in-memory data structure to accept
 *        foreign columnar.
 */
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <xgboost/base.h>
#include <xgboost/data.h>

#include <cmath>
#include <vector>
#include <algorithm>

#include "simple_csr_source.h"
#include "columnar.h"
#include "../common/math.h"
#include "../common/bitfield.h"
#include "../common/device_helpers.cuh"

namespace xgboost {
namespace data {

template <typename T>
__global__ void CountValidKernel(Columnar<T> const column,
                                 bool has_missing, float missing,
                                 int32_t* flag, common::Span<bst_row_t> offsets) {
  auto const tid =  threadIdx.x + blockDim.x * blockIdx.x;
  bool const missing_is_nan = common::CheckNAN(missing);

  if (tid >= column.size) {
    return;
  }
  RBitField8 const mask = column.valid;

  if (!has_missing) {
    if ((mask.Data() == nullptr || mask.Check(tid)) &&
        !common::CheckNAN(column.data[tid])) {
      offsets[tid+1] += 1;
    }
  } else if (missing_is_nan) {
    if (!common::CheckNAN(column.data[tid])) {
      offsets[tid+1] += 1;
    }
  } else {
    if (!common::CloseTo(column.data[tid], missing)) {
      offsets[tid+1] += 1;
    }
    if (common::CheckNAN(column.data[tid])) {
      *flag = 1;
    }
  }
}

template <typename T>
__device__ void AssignValue(T fvalue, int32_t colid,
                            common::Span<bst_row_t> out_offsets, common::Span<Entry> out_data) {
  auto const tid = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t oid = out_offsets[tid];
  out_data[oid].fvalue = fvalue;
  out_data[oid].index = colid;
  out_offsets[tid] += 1;
}

template <typename T>
__global__ void CreateCSRKernel(Columnar<T> const column,
                                int32_t colid, bool has_missing, float missing,
                                common::Span<bst_row_t> offsets, common::Span<Entry> out_data) {
  auto const tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (column.size <= tid) {
    return;
  }
  bool const missing_is_nan = common::CheckNAN(missing);
  if (!has_missing) {
    // no missing value is specified
    if ((column.valid.Data() == nullptr || column.valid.Check(tid)) &&
        !common::CheckNAN(column.data[tid])) {
      AssignValue(column.data[tid], colid, offsets, out_data);
    }
  } else if (missing_is_nan) {
    // specified missing value, but it's NaN
    if (!common::CheckNAN(column.data[tid])) {
      AssignValue(column.data[tid], colid, offsets, out_data);
    }
  } else {
    // specified missing value, and it's not NaN
    if (!common::CloseTo(column.data[tid], missing)) {
      AssignValue(column.data[tid], colid, offsets, out_data);
    }
  }
}

template <typename T>
void CountValid(std::vector<Json> const& j_columns, uint32_t column_id,
                bool has_missing, float missing,
                HostDeviceVector<bst_row_t>* out_offset,
                dh::caching_device_vector<int32_t>* out_d_flag,
                uint32_t* out_n_rows) {
  uint32_t constexpr kThreads = 256;
  auto const& j_column = j_columns[column_id];
  auto const& column_obj = get<Object const>(j_column);
  Columnar<T> foreign_column = ArrayInterfaceHandler::ExtractArray<T>(column_obj);
  uint32_t const n_rows = foreign_column.size;

  auto ptr = foreign_column.data.data();
  int32_t device = dh::CudaGetPointerDevice(ptr);
  CHECK_NE(device, -1);
  dh::safe_cuda(cudaSetDevice(device));

  if (column_id == 0) {
    out_offset->SetDevice(device);
    out_offset->Resize(n_rows + 1);
  }
  CHECK_EQ(out_offset->DeviceIdx(), device)
      << "All columns should use the same device.";
  CHECK_EQ(out_offset->Size(), n_rows + 1)
      << "All columns should have same number of rows.";

  common::Span<bst_row_t> s_offsets = out_offset->DeviceSpan();

  uint32_t const kBlocks = common::DivRoundUp(n_rows, kThreads);
  dh::LaunchKernel {kBlocks, kThreads} (
      CountValidKernel<T>,
      foreign_column,
      has_missing, missing,
      out_d_flag->data().get(), s_offsets);
  *out_n_rows = n_rows;
}

template <typename T>
void CreateCSR(std::vector<Json> const& j_columns, uint32_t column_id, uint32_t n_rows,
               bool has_missing, float missing,
               dh::device_vector<bst_row_t>* tmp_offset, common::Span<Entry> s_data) {
  uint32_t constexpr kThreads = 256;
  auto const& j_column = j_columns[column_id];
  auto const& column_obj = get<Object const>(j_column);
  Columnar<T> foreign_column = ArrayInterfaceHandler::ExtractArray<T>(column_obj);
  uint32_t kBlocks = common::DivRoundUp(n_rows, kThreads);
  dh::LaunchKernel {kBlocks, kThreads} (
      CreateCSRKernel<T>,
      foreign_column, column_id, has_missing, missing,
      dh::ToSpan(*tmp_offset), s_data);
}

void SimpleCSRSource::FromDeviceColumnar(std::vector<Json> const& columns,
                                         bool has_missing, float missing) {
  auto const n_cols = columns.size();
  int32_t constexpr kThreads = 256;

  dh::caching_device_vector<int32_t> d_flag;
  if (!common::CheckNAN(missing)) {
    d_flag.resize(1);
    thrust::fill(d_flag.begin(), d_flag.end(), 0);
  }
  uint32_t n_rows {0};
  for (size_t i = 0; i < n_cols; ++i) {
    auto const& typestr = get<String const>(columns[i]["typestr"]);
    DISPATCH_TYPE(CountValid, typestr,
                  columns, i, has_missing, missing, &(this->page_.offset), &d_flag, &n_rows);
  }
  // don't pay for what you don't use.
  if (!common::CheckNAN(missing)) {
    int32_t flag {0};
    dh::safe_cuda(cudaMemcpy(&flag, d_flag.data().get(), sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK_EQ(flag, 0) << "missing value is specifed but input data contains NaN.";
  }

  info.num_col_ = n_cols;
  info.num_row_ = n_rows;

  auto s_offsets = this->page_.offset.DeviceSpan();
  thrust::device_ptr<bst_row_t> p_offsets(s_offsets.data());
  CHECK_GE(s_offsets.size(), n_rows + 1);

  thrust::inclusive_scan(p_offsets, p_offsets + n_rows + 1, p_offsets);
  // Created for building csr matrix, where we need to change index after processing each
  // column.
  dh::device_vector<bst_row_t> tmp_offset(this->page_.offset.Size());
  dh::safe_cuda(cudaMemcpy(tmp_offset.data().get(), s_offsets.data(),
                           s_offsets.size_bytes(), cudaMemcpyDeviceToDevice));

  // We can use null_count from columnar data format, but that will add a non-standard
  // entry in the array interface, also involves accumulating from all columns.  Invoking
  // one copy seems easier.
  this->info.num_nonzero_ = tmp_offset.back();

  // Device is obtained and set in `CountValid'
  int32_t const device = this->page_.offset.DeviceIdx();
  this->page_.data.SetDevice(device);
  this->page_.data.Resize(this->info.num_nonzero_);
  auto s_data = this->page_.data.DeviceSpan();

  int32_t kBlocks = common::DivRoundUp(n_rows, kThreads);
  for (size_t i = 0; i < n_cols; ++i) {
    auto const& typestr = get<String const>(columns[i]["typestr"]);
    DISPATCH_TYPE(CreateCSR, typestr, columns, i, n_rows,
                  has_missing, missing, &tmp_offset, s_data);
  }
}

}  // namespace data
}  // namespace xgboost
