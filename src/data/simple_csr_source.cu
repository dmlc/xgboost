/*!
 * Copyright 2019 by XGBoost Contributors
 * \file simple_csr_source.cuh
 * \brief An extension for the simple CSR source in-memory data structure to accept
    foreign columnar data buffers, and convert them to XGBoost's internal DMatrix
 * \author Andrey Adinets
 * \author Matthew Jones
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
#include "../common/bitfield.cuh"
#include "../common/device_helpers.cuh"

namespace xgboost {
namespace data {

__global__ void CountValidKernel(BitField valid,
                                 foreign_size_type const n_rows,
                                 common::Span<size_t> offsets) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (n_rows <= tid) {
    return;
  }
  if (valid.Data() == nullptr || valid.Check(tid)) {
    ++offsets[tid + 1];
  }
}

__global__ void CreateCSRKernel(ForeignColumn col, int32_t col_idx,
                                ForeignCSR csr) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (col.data.size() <= tid) {
    return;
  }
  if (col.valid.Size() == 0 || col.valid.Check(tid)) {
    foreign_size_type oid = csr.offsets[tid];
    common::Span<float> d = col.data;
    csr.data[oid].fvalue = d[tid];
    csr.data[oid].index = col_idx;
  }
}

void ForeignColsToCSR(std::vector<ForeignColumn> const& cols, ForeignCSR& csr) {
  foreign_size_type n_rows = cols[0].size;
  foreign_size_type n_cols = cols.size();
  int32_t constexpr threads = 1024;
  int32_t const blocks = common::DivRoundUp(n_rows, threads);

  CHECK_GE(csr.offsets.size(), n_rows + 1);
  dh::safe_cuda(cudaMemset(csr.offsets.data(), 0, sizeof(foreign_size_type) * (n_rows + 1)));

  if (blocks > 0) {
    for (foreign_size_type i = 0; i < n_cols; ++i) {
      CountValidKernel<<<blocks, threads>>>(cols[i].valid, n_rows, csr.offsets);
    }

    thrust::device_ptr<size_t> p_offsets(csr.offsets.data());
    CHECK_GE(csr.offsets.size(), n_rows + 1);
    thrust::device_vector<size_t> offsets (n_rows + 1);

    thrust::inclusive_scan(p_offsets, p_offsets + n_rows + 1, p_offsets);

    csr.n_rows = n_rows;
    csr.n_cols = n_cols;

    for (foreign_size_type i = 0; i < n_cols; ++i) {
      dh::CudaCheckPointerDevice(csr.offsets.data());
      dh::CudaCheckPointerDevice(csr.data.data());
      CreateCSRKernel<<<blocks, threads>>>(cols[i], i, csr);
    }
  }
  dh::safe_cuda(cudaGetLastError());
  dh::safe_cuda(cudaDeviceSynchronize());
}

void SimpleCSRSource::CopyFrom(std::vector<ForeignColumn> cols) {
  size_t n_cols = cols.size();
  CHECK_GT(n_cols, 0);
  foreign_size_type n_valid = 0;  // total number of valid entries
  for (foreign_size_type i = 0; i < n_cols; ++i) {
    CHECK_EQ(cols[0].size, cols[i].size);
    cols[i].null_count = 0;
    dh::CudaCheckPointerDevice(cols[i].data.data());
    n_valid += cols[i].data.size() - cols[i].null_count;
  }

  info.num_col_     = n_cols;
  info.num_row_     = cols[0].size;
  info.num_nonzero_ = n_valid;

  GPUSet devices = GPUSet::Range(0, 1);

  page_.offset.Reshard(GPUDistribution(devices));
  page_.offset.Resize(info.num_row_ + 1);
  // page_.offset.Fill(1);
  dh::safe_cuda(cudaGetLastError());
  dh::safe_cuda(cudaDeviceSynchronize());

  page_.data.Reshard(GPUDistribution(devices));
  page_.data.Resize(n_valid);

  ForeignCSR csr;
  csr.data = page_.data.DeviceSpan(0);
  csr.offsets = page_.offset.DeviceSpan(0);

  ForeignColsToCSR(cols, csr);
  csr.n_nonzero = n_valid;
}

}  // namespace data
}  // namespace xgboost