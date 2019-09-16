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

template <typename T, size_t kBlockThreads>
__global__ void CountValidKernel(
    common::Span<Columnar<T> const> columns, int32_t const n_rows,
    bool has_missing, float missing, int32_t* flag, common::Span<size_t> offsets) {
  // One block for a column
  auto const bid = blockIdx.x;
  auto const tid =  threadIdx.x;
  bool const missing_is_nan = common::CheckNAN(missing);
  if (bid >= columns.size()) {
    return;
  }
  RBitField8 const mask = columns[bid].valid;

  if (!has_missing) {
    // no missing value is specified
    for (auto r = tid; r < n_rows; r += kBlockThreads) {
      if ((mask.Data() == nullptr || mask.Check(r)) &&
          !common::CheckNAN(columns[bid].data[r])) {
        atomicAdd(reinterpret_cast<BitFieldAtomicType*>(&offsets[r+1]),
                  static_cast<BitFieldAtomicType>(1));
      }
    }
  } else if (missing_is_nan) {
    // specifed missing, but it's NaN
    for (auto r = tid; r < n_rows; r += kBlockThreads) {
      if (!common::CheckNAN(columns[bid].data[r])) {
        atomicAdd(reinterpret_cast<BitFieldAtomicType*>(&offsets[r+1]),
                  static_cast<BitFieldAtomicType>(1));
      }
    }
  } else {
    // specified missing, and it's not NaN
    for (auto r = tid; r < n_rows; r += kBlockThreads) {
      if (!common::CloseTo(columns[bid].data[r], missing)) {
        atomicAdd(reinterpret_cast<BitFieldAtomicType*>(&offsets[r+1]),
                  static_cast<BitFieldAtomicType>(1));
      }
      if (common::CheckNAN(columns[bid].data[r])) {
        *flag = 1;
      }
    }
  }
}
template <typename T>
__global__ void CountValidKernel(Columnar<T> const column, int32_t const n_rows,
                                 bool has_missing, float missing, int32_t* flag,
                                 common::Span<size_t> offsets, uint32_t column_id) {
  auto const tid =  threadIdx.x + blockDim.x * blockIdx.x;
  bool const missing_is_nan = common::CheckNAN(missing);

  if (tid >= column.size) {
    return;
  }
  RBitField8 const mask = column.valid;

  if (!has_missing) {
    if ((mask.Data() == nullptr || mask.Check(tid)) &&
        !common::CheckNAN(column.data[tid])) {
      atomicAdd(reinterpret_cast<BitFieldAtomicType*>(&offsets[tid+1]),
                static_cast<BitFieldAtomicType>(1));
    }
  } else if (missing_is_nan) {
    if (!common::CheckNAN(column.data[tid])) {
      atomicAdd(reinterpret_cast<BitFieldAtomicType*>(&offsets[tid+1]),
                static_cast<BitFieldAtomicType>(1));
    }
  } else {
    if (!common::CloseTo(column.data[tid], missing)) {
      atomicAdd(reinterpret_cast<BitFieldAtomicType*>(&offsets[tid+1]),
                static_cast<BitFieldAtomicType>(1));
    }
    if (common::CheckNAN(column.data[tid])) {
      *flag = 1;
    }
  }
}

template <typename T>
__device__ void AssignValue(T fvalue, int32_t colid,
                            common::Span<size_t> out_offsets, common::Span<Entry> out_data) {
  auto const tid = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t oid = out_offsets[tid];
  out_data[oid].fvalue = fvalue;
  out_data[oid].index = colid;
  out_offsets[tid] += 1;
}

template <typename T>
__global__ void CreateCSRKernel(Columnar<T> const column,
                                int32_t colid, bool has_missing, float missing,
                                common::Span<size_t> offsets, common::Span<Entry> out_data) {
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
void FromDeviceColumnarImpl(std::vector<Json> const& columns,
                            bool has_missing, float missing,
                            SparsePage* p_page, MetaInfo* p_info) {
  size_t const n_columns = columns.size();
  std::vector<Columnar<T>> foreign_cols(n_columns);

  for (size_t i = 0; i < foreign_cols.size(); ++i) {
    CHECK(IsA<Object>(columns[i]));
    auto const& column = get<Object const>(columns[i]);
    foreign_cols[i] = ArrayInterfaceHandler::ExtractArray<T>(column);
  }

  p_info->num_col_ = n_columns;
  p_info->num_row_ = foreign_cols[0].size;

  uint64_t const n_cols = foreign_cols.size();
  uint64_t const n_rows = foreign_cols[0].size;

  auto ptr = foreign_cols[0].data.data();
  int32_t device = dh::CudaGetPointerDevice(ptr);
  CHECK_NE(device, -1);

  // validate correct ptr device
  for (int32_t i = 1; i < n_cols; ++i) {
    auto ptr = foreign_cols[i].data.data();
    int32_t ptr_device = dh::CudaGetPointerDevice(ptr);
    CHECK_EQ(device, ptr_device)
        << "GPU ID at 0^th column: " << device << ", "
        << "GPU ID at column " << i << ": " << ptr_device;
    CHECK_EQ(foreign_cols[0].size, foreign_cols[i].size)
        << "Columns should be of same size.";
  }

  dh::safe_cuda(cudaSetDevice(device));

  p_page->offset.SetDevice(device);
  p_page->offset.Resize(p_info->num_row_ + 1);

  auto s_offsets = p_page->offset.DeviceSpan();
  CHECK_EQ(s_offsets.size(), n_rows + 1);

  int32_t constexpr kThreads = 256;
  dh::device_vector<Columnar<T>> d_cols(foreign_cols);
  auto s_d_cols = dh::ToSpan(d_cols);

  dh::safe_cuda(cudaMemset(s_offsets.data(), 0, sizeof(int32_t) * (n_rows + 1)));
  dh::caching_device_vector<int32_t> d_flag;
  if (!common::CheckNAN(missing)) {
    d_flag.resize(1, 0);
  }

  for (int32_t i = 0; i < n_cols; ++i) {
    CountValidKernel<T><<<n_cols, kThreads>>>(s_d_cols, n_rows,
                                              has_missing, missing,
                                              d_flag.data().get(), s_offsets, i);
  }
  // don't pay for what you don't use.
  if (!common::CheckNAN(missing)) {
    int32_t flag {0};
    dh::safe_cuda(cudaMemcpy(&flag, d_flag.data().get(), sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK_EQ(flag, 0) << "missing value is specifed but input data contains NaN.";
  }

  thrust::device_ptr<size_t> p_offsets(s_offsets.data());
  CHECK_GE(s_offsets.size(), n_rows + 1);

  thrust::inclusive_scan(p_offsets, p_offsets + n_rows + 1, p_offsets);
  // Created for building csr matrix, where we need to change index after processing each
  // column.
  dh::device_vector<size_t> tmp_offset(p_page->offset.Size());
  dh::safe_cuda(cudaMemcpy(tmp_offset.data().get(), s_offsets.data(),
                           s_offsets.size_bytes(), cudaMemcpyDeviceToDevice));

  // We can use null_count from columnar data format, but that will add a non-standard
  // entry in the array interface, also involves accumulating from all columns.  Invoking
  // one copy seems easier.
  p_info->num_nonzero_ = tmp_offset.back();

  p_page->data.SetDevice(device);
  p_page->data.Resize(p_info->num_nonzero_);
  auto s_data = p_page->data.DeviceSpan();

  int32_t kBlocks = common::DivRoundUp(n_rows, kThreads);
  for (size_t col = 0; col < n_cols; ++col) {
    CreateCSRKernel<T><<<kBlocks, kThreads>>>(d_cols[col], col, has_missing, missing,
                                              dh::ToSpan(tmp_offset), s_data);
  }
}

template <typename T>
void CountValid(std::vector<Json> const& j_columns, uint32_t column_id,
                bool has_missing, float missing,
                SparsePage* p_page, dh::caching_device_vector<int32_t>* d_flag, uint32_t* out_n_rows) {
  int32_t constexpr kThreads = 256;
  auto const& j_column = j_columns[column_id];
  auto const& column_obj = get<Object const>(j_column);
  Columnar<T> foreign_column = ArrayInterfaceHandler::ExtractArray<T>(column_obj);
  uint32_t const n_rows = foreign_column.size;

  auto ptr = foreign_column.data.data();
  int32_t device = dh::CudaGetPointerDevice(ptr);
  CHECK_NE(device, -1);
  dh::safe_cuda(cudaSetDevice(device));

  p_page->offset.SetDevice(device);
  if (column_id == 0) {
    p_page->offset.Resize(n_rows + 1);
  }

  auto s_offsets = p_page->offset.DeviceSpan();
  CHECK_EQ(s_offsets.size(), n_rows + 1);

  int32_t kBlocks = common::DivRoundUp(n_rows, kThreads);
  CountValidKernel<T><<<kBlocks, kThreads>>>(
      foreign_column, n_rows,
      has_missing, missing,
      d_flag->data().get(), s_offsets, column_id);
  *out_n_rows = n_rows;
}

template <typename T>
void CreateCSR(std::vector<Json> const& j_columns, uint32_t column_id, uint32_t n_rows,
                        bool has_missing, float missing,
               dh::device_vector<size_t>* tmp_offset, common::Span<Entry> s_data) {
  int32_t constexpr kThreads = 256;
  auto const& j_column = j_columns[column_id];
  auto const& column_obj = get<Object const>(j_column);
  Columnar<T> foreign_column = ArrayInterfaceHandler::ExtractArray<T>(column_obj);
  int32_t kBlocks = common::DivRoundUp(n_rows, kThreads);
  CreateCSRKernel<T><<<kBlocks, kThreads>>>(foreign_column, column_id, has_missing, missing,
                                            dh::ToSpan(*tmp_offset), s_data);
}

void SimpleCSRSource::FromDeviceColumnar(std::vector<Json> const& columns,
                                         bool has_missing, float missing) {
  auto const& typestr = get<String const>(columns[0]["typestr"]);
  auto const n_cols = columns.size();
  int32_t constexpr kThreads = 256;

  uint32_t n_rows {0};
  dh::caching_device_vector<int32_t> d_flag;
  if (!common::CheckNAN(missing)) {
    d_flag.resize(1);
    thrust::fill(d_flag.begin(), d_flag.end(), 0);
  }
  DISPATCH_TYPE(CountValid, typestr,
                columns, 0, has_missing, missing, &(this->page_), &d_flag, &n_rows);
  for (size_t i = 1; i < n_cols; ++i) {
    uint32_t n_rows_i {0};
    DISPATCH_TYPE(CountValid, typestr,
                  columns, i, has_missing, missing, &(this->page_), &d_flag, &n_rows_i);
    CHECK_EQ(n_rows, n_rows_i) << "Each column should have same number of rows.";
  }
  // don't pay for what you don't use.
  if (!common::CheckNAN(missing)) {
    int32_t flag {0};
    dh::safe_cuda(cudaMemcpy(&flag, d_flag.data().get(), sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK_EQ(flag, 0) << "missing value is specifed but input data contains NaN.";
  }

  info.num_col_ = columns.size();
  info.num_row_ = n_rows;

  auto s_offsets = this->page_.offset.DeviceSpan();
  thrust::device_ptr<size_t> p_offsets(s_offsets.data());
  CHECK_GE(s_offsets.size(), n_rows + 1);

  thrust::inclusive_scan(p_offsets, p_offsets + n_rows + 1, p_offsets);
  // Created for building csr matrix, where we need to change index after processing each
  // column.
  dh::device_vector<size_t> tmp_offset(this->page_.offset.Size());
  dh::safe_cuda(cudaMemcpy(tmp_offset.data().get(), s_offsets.data(),
                           s_offsets.size_bytes(), cudaMemcpyDeviceToDevice));

  // We can use null_count from columnar data format, but that will add a non-standard
  // entry in the array interface, also involves accumulating from all columns.  Invoking
  // one copy seems easier.
  this->info.num_nonzero_ = tmp_offset.back();

  int device = this->page_.offset.DeviceIdx();
  this->page_.data.SetDevice(device);
  this->page_.data.Resize(this->info.num_nonzero_);
  auto s_data = this->page_.data.DeviceSpan();

  int32_t kBlocks = common::DivRoundUp(n_rows, kThreads);
  for (size_t i = 0; i < n_cols; ++i) {
    DISPATCH_TYPE(CreateCSR, typestr, columns, i, n_rows, has_missing, missing, &tmp_offset, s_data);
  }
}

}  // namespace data
}  // namespace xgboost
