/*!
 * Copyright 2018 by xgboost contributors
 */

#include <gdf/gdf.h>
#include <gdf/errorutils.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "../common/host_device_vector.h"
#include "../common/device_helpers.cuh"

#include "./gdf.cuh"
#include "./simple_csr_source.h"

#undef CUDA_TRY

#define CUDA_TRY(x) dh::safe_cuda(x)

namespace xgboost {
namespace data {

struct csr_gdf {
  Entry* data;
  size_t* offsets;
  size_t n_nz;
  size_t n_rows;
  size_t n_cols;
};

gdf_error run_converter(gdf_column** gdf_data, csr_gdf* csr);

//--- private CUDA functions / kernels
__global__ void cuda_create_csr_k
(void *gdf_data, gdf_valid_type* valid, gdf_dtype dtype, int col, Entry* data,
 gdf_size_type *offsets, size_t n_rows);

__global__ void determine_valid_rec_count_k
(gdf_valid_type* valid, size_t n_rows, size_t n_cols, size_t* offset);

__device__ int which_bitmap(int record) { return record / 8; }
__device__ int which_bit(int bit) { return bit % 8; }
__device__ int check_bit(gdf_valid_type data, int bit) {
  gdf_valid_type bit_mask[8] = {1, 2, 4, 8, 16, 32, 64, 128};
  return data & bit_mask[bit];
}

__device__ bool is_valid(gdf_valid_type* valid, int tid) {
  if (valid == nullptr)
    return true;
  int bitmap_idx = which_bitmap(tid);
  int bit_idx = which_bit(tid);
  gdf_valid_type bitmap = valid[bitmap_idx];
  return check_bit(bitmap, bit_idx);
}

//
//------------------------------------------------------------
//

/*
 * Convert a GDF into a CSR GDF
 *
 * Restrictions:  All columns need to be of the same length
 */
gdf_error gdf_to_csr(gdf_column** gdf_data, int n_cols, csr_gdf* csr) {
  gdf_error status = gdf_error::GDF_SUCCESS;
  size_t n_rows =  gdf_data[0]->size;

  //--------------------------------------------------------------------------------------
  // The first step is to create an array that counts the number of valid entries per row
  // this is done by each thread looking across its row and checking the valid bits

  //-- threads and blocks
  int threads = 1024;
  int blocks = (n_rows + threads - 1) / threads;

  // Allocate space for the offset - this will eventually be IA -
  // dtype is long since the sum of all column elements could be larger than int32
  //gdf_size_type * offsets;
  //CUDA_TRY(cudaMalloc(&offsets, (numRows + 2) * sizeof(gdf_size_type)));
  size_t* offsets = csr->offsets;
  //CUDA_TRY(cudaMemset(offsets, 0, sizeof(gdf_size_type) * (numRows + 2)));
  CUDA_TRY(cudaMemset(offsets, 0, sizeof(gdf_size_type) * (n_rows + 1)));

  if (blocks > 0) {
    for (int i = 0; i < n_cols; ++i) {
      determine_valid_rec_count_k<<<blocks, threads>>>
        (gdf_data[i]->valid, n_rows, n_cols, offsets);
      CUDA_TRY(cudaGetLastError());
      CUDA_TRY(cudaDeviceSynchronize());
    }
  }

  //--------------------------------------------------------------------------------------
  // compute the number of elements
  thrust::device_ptr<size_t> offsets_begin(offsets);
  int64_t n_elements = thrust::reduce
    (offsets_begin, offsets_begin + n_rows, 0ull, thrust::plus<size_t>());

  //--------------------------------------------------------------------------------------
  // Now do an exclusive scan to compute the offsets for where to write data
  thrust::exclusive_scan(offsets_begin, offsets_begin + n_rows + 1, offsets_begin);

  //----------------------------------------------------------------------------------
  // Now its time to start copying data over
  //gdf_size_type *   JA;
  //CUDA_TRY(cudaMalloc(&JA, (sizeof(gdf_size_type) * numElements)));
  //----------------------------------------------------------------------------------
  // Now just missing A and the moving of data

  csr->n_rows = n_rows;
  csr->n_cols = n_cols;
  csr->n_nz = n_elements;

  // Start processing based on data type
  status = run_converter(gdf_data, csr);
  return status;
}

gdf_error run_converter(gdf_column** gdf_data, csr_gdf* csr) {
  size_t n_cols = csr->n_cols;
  size_t n_rows = csr->n_rows;
  
  int threads = 256;
  int blocks = (n_rows + threads - 1) / threads;

  //T *  A;
  //CUDA_TRY(cudaMalloc(&A, (sizeof(T) * csrReturn->nnz)));
  //CUDA_TRY(cudaMemset(A, 0, (sizeof(T) * csrReturn->nnz)));
  // temporary offsets for writing data
  thrust::device_ptr<size_t> offset_begin(csr->offsets);
  thrust::device_vector<size_t> offsets2(offset_begin, offset_begin + n_rows + 1);

  // Now start moving the data and creating the CSR
  if (blocks > 0) {
    for (int col = 0; col < n_cols; ++col) {
      gdf_column *gdf = gdf_data[col];
      cuda_create_csr_k<<<blocks, threads>>>
        (gdf->data, gdf->valid, gdf->dtype, col, csr->data,
         offsets2.data().get(), n_rows);
      CUDA_TRY(cudaGetLastError());
    }
  }
  return gdf_error::GDF_SUCCESS;
}

/*
 * Move data over into CSR and possible convert format
 */
__global__ void cuda_create_csr_k
(void* gdf_data, gdf_valid_type* valid, gdf_dtype dtype, int col,
 Entry* data, size_t* offsets, size_t n_rows) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;      // get the tread ID which is also the row number
  if (tid >= n_rows)
    return;
  gdf_size_type offset_idx = offsets[tid];              // where should this thread start writing data
  if (is_valid(valid, tid)) {
    data[offset_idx].fvalue = convert_data_element(gdf_data, tid, dtype);
    data[offset_idx].index = col;
    ++offsets[tid];
  }
}

/*
 * Compute the number of valid entries per rows - a row spans multiple gdf_colums -
 * There is one thread running per row, so just compute the sum for this row.
 *
 * the number of elements a valid array is actually ceil(numRows / 8) since it is a bitmap. 
 * the total number of bits checked is equal to numRows
 *
 */
__global__ void determine_valid_rec_count_k
(gdf_valid_type *valid, size_t n_rows, size_t n_cols, gdf_size_type* offset) {

  int tid = threadIdx.x + blockDim.x * blockIdx.x;        // get the tread ID which is also the row number
  if (tid >= n_rows)
    return;
  if (is_valid(valid, tid))
    ++offset[tid];
}

void SimpleCSRSource::InitFromGDF(gdf_column** cols, size_t n_cols) {
  CHECK_GT(n_cols, 0);
  size_t n_rows = cols[0]->size;
  info.num_col_ = n_cols;
  info.num_row_ = n_rows;
  size_t n_entries = 0;
  for (size_t i = 0; i < n_cols; ++i) {
    CHECK_EQ(n_rows, cols[i]->size);
    n_entries += cols[i]->size - cols[i]->null_count;
  }
  info.num_nonzero_ = n_entries;
  // TODO(canonizer): use the same devices as by the rest of xgboost
  GPUSet devices = GPUSet::Range(0, 1);
  page_.offset.Reshard(GPUDistribution::Overlap(devices, 1));
  // TODO(canonizer): use the real row offsets for the multi-GPU case
  std::vector<size_t> device_offsets{0, n_entries};
  page_.data.Reshard(GPUDistribution::Explicit(devices, device_offsets));
  page_.offset.Resize(n_rows + 1);
  page_.data.Resize(n_entries);
  csr_gdf csr;
  csr.data = page_.data.DevicePointer(0);
  csr.offsets = page_.offset.DevicePointer(0);
  csr.n_nz = 0;
  csr.n_rows = n_rows;
  csr.n_cols = n_cols;
  gdf_error status = gdf_to_csr(cols, n_cols, &csr);
  CHECK_EQ(status, gdf_error::GDF_SUCCESS);
  //info.num_nonzero_ = csr.n_nz;
}

}  // namespace data
}  // namespace xgboost
