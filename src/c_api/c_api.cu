// Copyright (c) 2019 by Contributors

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <xgboost/c_api.h>
#include <xgboost/data.h>
#include <memory>

#include "../common/device_helpers.cuh"
#include "../common/host_device_vector.h"
#include "../data/simple_csr_source.h"
#include "./c_api_error.h"

namespace xgboost {
typedef unsigned char cudf_interchange_valid_type;
typedef int32_t
    cudf_interchange_size_type; /**< Limits the maximum size of a
                                   cudf_interchange_column to 2^31-1 */
typedef enum {
  invalid = 0,
  INT8,
  INT16,
  INT32,
  INT64,
  FLOAT32,
  FLOAT64,
} cudf_interchange_dtype;

/** We define a simple interchange format loosely based on the internal column
 * structure of cudf (cuda data frame). cudf is responsible for passing this
 * exact structure to the xgboost C API for DMatrix construction. The decoupled
 * interchange format means xgboost does not depend on the specific internal
 * structure of cudf which may change. */

typedef struct cudf_interchange_column_ {
  void* data; /**< Pointer to the columns data */
  cudf_interchange_valid_type*
      valid; /**< Pointer to the columns validity bit mask where the
  'i'th bit indicates if the 'i'th row is NULL */
  cudf_interchange_size_type size; /**< Number of data elements in the columns
               data buffer. Limited to 2^31 - 1.*/
  cudf_interchange_dtype dtype;    /**< The datatype of the column's data */
  int32_t null_count; /**< The number of NULL values in the column's data */
  char* col_name;     // host-side:	null terminated string
  cudf_interchange_column_() {
    static_assert(sizeof(cudf_interchange_column_) == 40,
                  "If this static assert fails, the compiler is not supported "
                  "- please file an issue");
  }
} cudf_interchange_column;

struct CsrCudf {
  Entry* data;
  size_t* offsets;
  size_t n_nz;
  size_t n_rows;
  size_t n_cols;
};

/**
 * Convert the data element into a common format
 */
__device__ inline float ConvertDataElement(void* data, int tid,
                                           cudf_interchange_dtype dtype) {
  switch (dtype) {
    case cudf_interchange_dtype::INT8: {
      int8_t* d = (int8_t*)data;
      return float(d[tid]);
    }
    case cudf_interchange_dtype::INT16: {
      int16_t* d = (int16_t*)data;
      return float(d[tid]);
    }
    case cudf_interchange_dtype::INT32: {
      int32_t* d = (int32_t*)data;
      return float(d[tid]);
    }
    case cudf_interchange_dtype::INT64: {
      int64_t* d = (int64_t*)data;
      return float(d[tid]);
    }
    case cudf_interchange_dtype::FLOAT32: {
      float* d = (float*)data;
      return float(d[tid]);
    }
    case cudf_interchange_dtype::FLOAT64: {
      double* d = (double*)data;
      return float(d[tid]);
    }
  }
  return nanf(nullptr);
}

void RunConverter(cudf_interchange_column** gdf_data, CsrCudf* csr);

//--- private CUDA functions / kernels
__global__ void cuda_create_csr_k(void* cudf_data,
                                  cudf_interchange_valid_type* valid,
                                  cudf_interchange_dtype dtype, int col,
                                  Entry* data, size_t* offsets, size_t n_rows);

__global__ void determine_valid_rec_count_k(cudf_interchange_valid_type* valid,
                                            size_t n_rows, size_t n_cols,
                                            size_t* offset);

__device__ int WhichBitmap(int record) { return record / 8; }
__device__ int WhichBit(int bit) { return bit % 8; }
__device__ int CheckBit(cudf_interchange_valid_type data, int bit) {
  cudf_interchange_valid_type bit_mask[8] = {1, 2, 4, 8, 16, 32, 64, 128};
  return data & bit_mask[bit];
}

__device__ bool IsValid(cudf_interchange_valid_type* valid, int tid) {
  if (valid == nullptr) return true;
  int bitmap_idx = WhichBitmap(tid);
  int bit_idx = WhichBit(tid);
  cudf_interchange_valid_type bitmap = valid[bitmap_idx];
  return CheckBit(bitmap, bit_idx);
}

// Convert a CUDF into a CSR CUDF
void CUDFToCSR(cudf_interchange_column** cudf_data, int n_cols, CsrCudf* csr) {
  size_t n_rows = cudf_data[0]->size;

  // the first step is to create an array that counts the number of valid
  // entries per row this is done by each thread looking across its row and
  // checking the valid bits
  int threads = 1024;
  int blocks = (n_rows + threads - 1) / threads;

  size_t* offsets = csr->offsets;
  dh::safe_cuda(cudaMemset(offsets, 0,
                           sizeof(cudf_interchange_size_type) * (n_rows + 1)));

  if (blocks > 0) {
    for (int i = 0; i < n_cols; ++i) {
      determine_valid_rec_count_k<<<blocks, threads>>>(cudf_data[i]->valid,
                                                       n_rows, n_cols, offsets);
      dh::safe_cuda(cudaGetLastError());
      dh::safe_cuda(cudaDeviceSynchronize());
    }
  }

  // compute the number of elements
  thrust::device_ptr<size_t> offsets_begin(offsets);
  int64_t n_elements = thrust::reduce(offsets_begin, offsets_begin + n_rows,
                                      0ull, thrust::plus<size_t>());

  // now do an exclusive scan to compute the offsets for where to write data
  thrust::exclusive_scan(offsets_begin, offsets_begin + n_rows + 1,
                         offsets_begin);

  csr->n_rows = n_rows;
  csr->n_cols = n_cols;
  csr->n_nz = n_elements;

  // process based on data type
  RunConverter(cudf_data, csr);
}

void RunConverter(cudf_interchange_column** cudf_data, CsrCudf* csr) {
  size_t n_cols = csr->n_cols;
  size_t n_rows = csr->n_rows;

  int threads = 256;
  int blocks = (n_rows + threads - 1) / threads;

  // temporary offsets for writing data
  thrust::device_ptr<size_t> offset_begin(csr->offsets);
  thrust::device_vector<size_t> offsets2(offset_begin,
                                         offset_begin + n_rows + 1);

  // move the data and create the CSR
  if (blocks > 0) {
    for (int col = 0; col < n_cols; ++col) {
      cudf_interchange_column* cudf = cudf_data[col];
      cuda_create_csr_k<<<blocks, threads>>>(cudf->data, cudf->valid,
                                             cudf->dtype, col, csr->data,
                                             offsets2.data().get(), n_rows);
      dh::safe_cuda(cudaGetLastError());
    }
  }
}

// move data over into CSR and possibly convert the format
__global__ void cuda_create_csr_k(void* cudf_data,
                                  cudf_interchange_valid_type* valid,
                                  cudf_interchange_dtype dtype, int col,
                                  Entry* data, size_t* offsets, size_t n_rows) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= n_rows) return;
  cudf_interchange_size_type offset_idx = offsets[tid];
  if (IsValid(valid, tid)) {
    data[offset_idx].fvalue = ConvertDataElement(cudf_data, tid, dtype);
    data[offset_idx].index = col;
    ++offsets[tid];
  }
}

// compute the number of valid entries per row
__global__ void determine_valid_rec_count_k(
    cudf_interchange_valid_type* valid, size_t n_rows, size_t n_cols,
    cudf_interchange_size_type* offset) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= n_rows) return;
  if (IsValid(valid, tid)) ++offset[tid];
}

void InitFromCUDF(data::SimpleCSRSource* source, cudf_interchange_column** cols,
                  size_t n_cols) {
  CHECK_GT(n_cols, 0);
  size_t n_rows = cols[0]->size;
  source->info.num_col_ = n_cols;
  source->info.num_row_ = n_rows;
  size_t n_entries = 0;
  for (size_t i = 0; i < n_cols; ++i) {
    CHECK_EQ(n_rows, cols[i]->size);
    n_entries += cols[i]->size - cols[i]->null_count;
  }
  source->info.num_nonzero_ = n_entries;
  // TODO(canonizer): use the same devices as by the rest of xgboost
  GPUSet devices = GPUSet::Range(0, 1);
  source->page_.offset.Reshard(GPUDistribution::Overlap(devices, 1));
  // TODO(canonizer): use the real row offsets for the multi-GPU case
  std::vector<size_t> device_offsets{0, n_entries};
  source->page_.data.Reshard(
      GPUDistribution::Explicit(devices, device_offsets));
  source->page_.offset.Resize(n_rows + 1);
  source->page_.data.Resize(n_entries);
  CsrCudf csr;
  csr.data = source->page_.data.DevicePointer(0);
  csr.offsets = source->page_.offset.DevicePointer(0);
  csr.n_nz = 0;
  csr.n_rows = n_rows;
  csr.n_cols = n_cols;
  CUDFToCSR(cols, n_cols, &csr);
}

int XGDMatrixCreateFromCUDF(void** cols, size_t n_cols, DMatrixHandle* out) {
  API_BEGIN();
  std::unique_ptr<data::SimpleCSRSource> source(new data::SimpleCSRSource());
  InitFromCUDF(source.get(), reinterpret_cast<cudf_interchange_column**>(cols),
               n_cols);
  *out = new std::shared_ptr<DMatrix>(DMatrix::Create(std::move(source)));
  API_END();
}

__global__ void unpack_cudf_column_k(float* data, size_t n_rows, size_t n_cols,
                                     cudf_interchange_column col) {
  size_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n_rows) return;
  data[n_cols * i] = ConvertDataElement(col.data, i, col.dtype);
}

void SetCUDFInfo(MetaInfo* info, const char* key,
                 cudf_interchange_column** cols, size_t n_cols) {
  CHECK_GT(n_cols, 0);
  size_t n_rows = cols[0]->size;
  for (size_t i = 0; i < n_cols; ++i) {
    CHECK_EQ(cols[i]->null_count, 0) << "all labels and weights must be valid";
    CHECK_EQ(cols[i]->size, n_rows)
        << "all CUDF columns must be of the same size";
  }
  HostDeviceVector<bst_float>* field = nullptr;
  if (!strcmp(key, "label")) {
    field = &info->labels_;
  } else if (!strcmp(key, "weight")) {
    field = &info->weights_;
    CHECK_EQ(n_cols, 1) << "only one CUDF column allowed for weights";
  } else {
    LOG(WARNING) << key << ": invalid key value for MetaInfo field";
    return;
  }
  // TODO(canonizer): use the same devices as elsewhere in xgboost
  GPUSet devices = GPUSet::Range(0, 1);
  field->Reshard(GPUDistribution::Granular(devices, n_cols));
  field->Resize(n_cols * n_rows);
  bst_float* data = field->DevicePointer(0);
  for (size_t i = 0; i < n_cols; ++i) {
    int block = 256;
    unpack_cudf_column_k<<<dh::DivRoundUp(n_rows, block), block>>>(
        data + i, n_rows, n_cols, *cols[i]);
    dh::safe_cuda(cudaGetLastError());
  }
}

XGB_DLL int XGDMatrixSetCUDFInfo(DMatrixHandle handle, const char* field,
                                 void** cols, size_t n_cols) {
  API_BEGIN();
  CHECK_HANDLE();
  MetaInfo* info =
      &static_cast<std::shared_ptr<DMatrix>*>(handle)->get()->Info();
  SetCUDFInfo(info, field, reinterpret_cast<cudf_interchange_column**>(cols),
              n_cols);
  API_END();
}
}  // namespace xgboost
