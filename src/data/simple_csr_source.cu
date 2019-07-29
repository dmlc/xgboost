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

namespace xgboost {
namespace data {

__device__ int which_bit (int bit) {
  return bit % 8;
}
__device__ int which_bitmap (int record) {
  return record / 8;
}

__device__ int check_bit (foreign_valid_type bitmap, int bid) {
  foreign_valid_type bitmask[8] = {1, 2, 4, 8, 16, 32, 64, 128};
  return bitmap & bitmask[bid];
}

__device__ bool is_valid(foreign_valid_type * valid, int tid) {
  if (valid == nullptr) {
    return true;
  }
  int bmid = which_bitmap(tid);
  int bid = which_bit(tid);
  foreign_valid_type bitmap = valid[bmid];
  return check_bit(bitmap, bid);
}

__global__ void CountValid(foreign_valid_type * valid,
                           foreign_size_type n_rows,
                           foreign_size_type n_cols,
                           size_t * offsets) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (n_rows <= tid) {
    return;
  } else if (is_valid(valid, tid)) {
    ++offsets[tid];
  }
}

__global__ void CreateCSR(ForeignColumn * col, int col_idx, ForeignCSR * csr) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (col->size <= tid) {
    return;
  } else if (is_valid(col->valid, tid)) {
    foreign_size_type oid = csr->offsets[tid];
    float * d = (float *) (col->data);
    csr->data[oid].fvalue = float(d[tid]);
    csr->data[oid].index = col_idx;
    ++csr->offsets[tid];
  }
}

void ForeignColsToCSR(ForeignColumn ** cols, foreign_size_type n_cols, ForeignCSR * csr) {
  foreign_size_type n_rows = cols[0]->size;
  int threads = 1024;
  int blocks = (n_rows + threads - 1) / threads;

  dh::safe_cuda(cudaMemset(csr->offsets, 0, sizeof(foreign_size_type) * (n_rows + 1)));
  if (0 < blocks) {
    for (foreign_size_type i = 0 ; i < n_cols; ++i) {
      CountValid <<<blocks, threads>>> (cols[i]->valid, n_rows, n_cols, csr->offsets);
      dh::safe_cuda(cudaGetLastError());
      dh::safe_cuda(cudaDeviceSynchronize());
    }

    thrust::device_ptr<size_t> offsets(csr->offsets);
    int64_t n_valid = thrust::reduce(offsets, offsets + n_rows, 0ull, thrust::plus<size_t>());
    thrust::exclusive_scan(offsets, offsets + n_rows + 1, offsets);

    csr->n_nonzero = n_valid;
    csr->n_rows = n_rows;
    csr->n_cols = n_cols;

    for (foreign_size_type i = 0; i < n_cols; ++i) {
      CreateCSR <<<blocks, threads>>> (cols[i], i, csr);
    }
  }
}

void SimpleCSRSource::CopyFrom(ForeignColumn ** cols, foreign_size_type n_cols) {
  CHECK_GT(n_cols, 0);
  foreign_size_type n_valid = 0;
  for (foreign_size_type i = 0; i < n_cols; ++i) {
    CHECK_EQ(cols[0]->size, cols[i]->size);
    n_valid += cols[i]->size - cols[i]->null_count;
  }

  info.num_col_ = n_cols;
  info.num_row_ = cols[0]->size;
  info.num_nonzero_ = n_valid;

  GPUSet devices = GPUSet::Range(0, 1);
  page_.offset.Reshard(GPUDistribution::Overlap(devices, 1));
  page_.offset.Resize(cols[0]->size + 1);

  std::vector<size_t> device_offsets{0, (size_t) n_valid};
  page_.data.Reshard(GPUDistribution::Explicit(devices, device_offsets));
  page_.data.Reshard(GPUDistribution::Overlap(devices, 1));
  page_.data.Resize(n_valid);

  ForeignCSR csr;
  csr.data = page_.data.DevicePointer(0);
  csr.offsets = page_.offset.DevicePointer(0);

  ForeignColsToCSR(cols, n_cols, &csr);
}

}  // namespace data
}  // namespace xgboost
