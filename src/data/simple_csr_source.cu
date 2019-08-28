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
  CountValidKernel<T, kThreads><<<n_cols, kThreads>>>(s_d_cols, n_rows,
                                                      has_missing, missing,
                                                      d_flag.data().get(), s_offsets);
  // don't pay for what you don't use.
  if (!common::CheckNAN(missing)) {
    int32_t flag {0};
    thrust::copy(d_flag.begin(), d_flag.end(), &flag);
    CHECK_EQ(flag, 0) << "missing value is specifed but input data contains NaN.";
  }

  thrust::device_ptr<size_t> p_offsets(s_offsets.data());
  CHECK_GE(s_offsets.size(), n_rows + 1);

  thrust::inclusive_scan(p_offsets, p_offsets + n_rows + 1, p_offsets);
  // Created for building csr matrix, where we need to change index after processing each
  // column.
  dh::device_vector<size_t> tmp_offset(p_page->offset.Size());
  thrust::copy(p_offsets, p_offsets + n_rows + 1, tmp_offset.begin());

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

void SimpleCSRSource::FromDeviceColumnar(std::vector<Json> const& columns,
                                         bool has_missing, float missing) {
  auto const& typestr = get<String const>(columns[0]["typestr"]);
  if (typestr.at(1) == 'f' && typestr.at(2) == '4') {
    FromDeviceColumnarImpl<float>(columns, has_missing, missing, &(this->page_), &(this->info));
  } else if (typestr.at(1) == 'f' && typestr.at(2) == '8') {
    FromDeviceColumnarImpl<double>(columns, has_missing, missing, &(this->page_), &(this->info));
  } else if (typestr.at(1) == 'i' && typestr.at(2) == '1') {
    FromDeviceColumnarImpl<int8_t>(columns, has_missing, missing, &(this->page_), &(this->info));
  } else if (typestr.at(1) == 'i' && typestr.at(2) == '2') {
    FromDeviceColumnarImpl<int16_t>(columns, has_missing, missing, &(this->page_), &(this->info));
  } else if (typestr.at(1) == 'i' && typestr.at(2) == '4') {
    FromDeviceColumnarImpl<int32_t>(columns, has_missing, missing, &(this->page_), &(this->info));
  } else if (typestr.at(1) == 'i' && typestr.at(2) == '8') {
    FromDeviceColumnarImpl<int64_t>(columns, has_missing, missing, &(this->page_), &(this->info));
  } else if (typestr.at(1) == 'u' && typestr.at(2) == '1') {
    FromDeviceColumnarImpl<uint8_t>(columns, has_missing, missing, &(this->page_), &(this->info));
  } else if (typestr.at(1) == 'u' && typestr.at(2) == '2') {
    FromDeviceColumnarImpl<uint16_t>(columns, has_missing, missing, &(this->page_), &(this->info));
  } else if (typestr.at(1) == 'u' && typestr.at(2) == '4') {
    FromDeviceColumnarImpl<uint32_t>(columns, has_missing, missing, &(this->page_), &(this->info));
  } else if (typestr.at(1) == 'u' && typestr.at(2) == '8') {
    FromDeviceColumnarImpl<uint64_t>(columns, has_missing, missing, &(this->page_), &(this->info));
  } else {
    LOG(FATAL) << ColumnarErrors::UnknownTypeStr(typestr);
  }
}

}  // namespace data
}  // namespace xgboost
