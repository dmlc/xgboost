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
#include <vector>
#include <algorithm>

#include "simple_csr_source.h"
#include "columnar.h"
#include "../common/bitfield.h"
#include "../common/device_helpers.cuh"

namespace xgboost {
namespace data {

template <size_t kBlockThreads>
__global__ void CountValidKernel(common::Span<Columnar const> columns,
                                 int32_t const n_rows,
                                 common::Span<size_t> offsets) {
  // One block for a column
  auto const bid = blockIdx.x;
  auto const tid =  threadIdx.x;
  if (bid >= columns.size()) {
    return;
  }
  RBitField8 const mask = columns[bid].valid;
  for (auto r = tid; r < n_rows; r += kBlockThreads) {
    if (mask.Data() == nullptr || mask.Check(r)) {
      atomicAdd(reinterpret_cast<BitFieldAtomicType*>(&offsets[r+1]),
                static_cast<BitFieldAtomicType>(1));
    }
  }
}

__global__ void CreateCSRKernel(Columnar const column,
                                int32_t colid,
                                common::Span<size_t> offsets,
                                common::Span<Entry> out_data) {
  auto tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (column.size <= tid) {
    return;
  }

  if (column.valid.Data() == nullptr || column.valid.Check(tid)) {
      int32_t oid = offsets[tid];
      out_data[oid].fvalue = column.data[tid];
      out_data[oid].index = colid;
      offsets[tid] += 1;
  }
}

void SimpleCSRSource::FromDeviceColumnar(std::vector<Columnar> cols) {
  uint64_t const n_cols = cols.size();
  uint64_t const n_rows = cols[0].size;

  auto ptr = cols[0].data.data();
  int32_t device = dh::CudaGetPointerDevice(ptr);
  CHECK_NE(device, -1);

  for (int32_t i = 1; i < n_cols; ++i) {
    auto ptr = cols[i].data.data();
    int32_t ptr_device = dh::CudaGetPointerDevice(ptr);
    CHECK_EQ(device, ptr_device)
        << "GPU ID at 0^th column: " << device << ", "
        << "GPU ID at column " << i << ": " << ptr_device;
  }

  dh::safe_cuda(cudaSetDevice(device));

  page_.offset.SetDevice(device);
  page_.offset.Resize(info.num_row_ + 1);

  page_.data.SetDevice(device);
  page_.data.Resize(info.num_nonzero_);

  auto s_data = page_.data.DeviceSpan();
  auto s_offsets = page_.offset.DeviceSpan();
  CHECK_EQ(s_offsets.size(), n_rows + 1);

  int32_t constexpr kThreads = 256;
  dh::device_vector<Columnar> d_cols(cols);
  auto s_d_cols = dh::ToSpan(d_cols);

  dh::safe_cuda(cudaMemset(s_offsets.data(), 0, sizeof(int32_t) * (n_rows + 1)));

  CountValidKernel<kThreads><<<n_cols, kThreads>>>(s_d_cols, n_rows, s_offsets);

  thrust::device_ptr<size_t> p_offsets(s_offsets.data());
  CHECK_GE(s_offsets.size(), n_rows + 1);

  thrust::inclusive_scan(p_offsets, p_offsets + n_rows + 1, p_offsets);
  // Created for building csr matrix, where we need to change index
  // after processing each column.
  dh::device_vector<size_t> tmp_offset(page_.offset.Size());
  thrust::copy(p_offsets, p_offsets + n_rows + 1, tmp_offset.begin());

  int32_t kBlocks = common::DivRoundUp(n_rows, kThreads);

  for (size_t col = 0; col < n_cols; ++col) {
    CreateCSRKernel<<<kBlocks, kThreads>>>(d_cols[col], col, dh::ToSpan(tmp_offset), s_data);
  }
}

}  // namespace data
}  // namespace xgboost
