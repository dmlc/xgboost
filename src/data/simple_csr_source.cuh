/*!
 * Copyright 2019 by XGBoost Contributors
 * \file simple_csr_source.cuh
 * \brief An extension for the simple CSR source in-memory data structure to accept
    foreign columnar data buffers, and convert them to XGBoost's internal DMatrix
 * \author Andrey Adinets
 * \author Matthew Jones
 */
#ifndef XGBOOST_DATA_SIMPLE_CSR_SOURCE_CUH_
#define XGBOOST_DATA_SIMPLE_CSR_SOURCE_CUH_

#include <xgboost/base.h>
#include <xgboost/data.h>
#include <vector>
#include <algorithm>

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

__device__ bool is_valid(foreign_valid_type * mask, int tid) {
  if (mask == nullptr) {
    return true;
  }
  int bmid = which_bitmap(tid);
  int bid = which_bit(tid);
  foreign_valid_type bitmap = mask[bmid];
  return check_bit(bitmap, bid)
}

__global__ void CountValid(void * data,
                           foreign_size_type n_rows,
                           foreign_size_type n_cols,
                           size_t * offsets) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (n_rows <= tid) {
    return;
  } else if (is_valid(mask, tid)) {
    ++offsets[tid];
  }
}

__global__ void CreateCSR(ForeignColumn * col, int col_idx, ForeignCSR * csr) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (col->size <= tid) {
    return;
  } else if (is_valid(col->valid, tid)) {
    foreign_size_type oid = csr->offsets[tid];
    csr->data[oid].fvalue = float(col->data[tid]);
    csr->data[oid].index = col_idx;
    ++csr->offsets][tid]
  }
}

void ForeignColsToCSR(ForeignColumn ** cols, foreign_size_type n_cols) {
  foreign_size_type n_rows = cols[0]->size;
  int threads = 1024;
  int blocks = (n_rows + threads - 1) / threads;

  dh::safe_cuda(cudaMemset(csr->offsets, 0, sizeof(foreign_size_type) * (n_rows + 1)));
  if (0 < blocks) {
    for (foreign_size_type i = 0 ; i < n_cols; ++i) {
      CountValid <<<blocks, threads>>> (cols[i]->data, n_rows, n_cols, csr->offsets);
      dh::safe_cuda(cudaGetLastError());
      dh::safe_cuda(cudaDeviceSynchronize());
    }

    thrust::device_ptr<size_t> offsets(csr->offsets);
    int64_t n_valid = thrust::reduce(offsets, offsets + n_rows, 0ull, thrust::plus<size_t>());
    thrust::exclusive_scan(offsets, offsets + n_rows + 1, offsets);

    ForeignCSR * csr;
    csr->n_nonzero = n_valid;
    csr->n_rows = n_rows;
    csr->n_cols = n_cols;

    for (foreign_size_type i; i < n_cols, ++i) {
      CreateCSR <<<blocks, threads>>> (cols[i], i, csr)
    }
  }

}
}  // namespace data
}  // namespace xgboost