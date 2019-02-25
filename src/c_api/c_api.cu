// Copyright (c) 2019 by Contributors

#include <xgboost/c_api.h>
#include <memory>
#include <xgboost/data.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "./c_api_error.h"
#include "../data/simple_csr_source.h"
#include "../common/host_device_vector.h"
#include "../common/device_helpers.cuh"



namespace xgboost {
typedef unsigned char gdf_valid_type;
typedef int gdf_size_type; /**< Limits the maximum size of a gdf_column to 2^31-1 */
typedef enum {
    GDF_invalid=0,
    GDF_INT8,
    GDF_INT16,
    GDF_INT32,
    GDF_INT64,
    GDF_FLOAT32,
    GDF_FLOAT64,
    GDF_DATE32,   	/**< int32_t days since the UNIX epoch */
    GDF_DATE64,   	/**< int64_t milliseconds since the UNIX epoch */
    GDF_TIMESTAMP,	/**< Exact timestamp encoded with int64 since UNIX epoch (Default unit millisecond) */
    GDF_CATEGORY,
    GDF_STRING,
    N_GDF_TYPES, 	/* additional types should go BEFORE N_GDF_TYPES */
} gdf_dtype;

typedef enum {
	TIME_UNIT_NONE=0, // default (undefined)
	TIME_UNIT_s,   // second
	TIME_UNIT_ms,  // millisecond
	TIME_UNIT_us,  // microsecond
	TIME_UNIT_ns   // nanosecond
} gdf_time_unit;

typedef struct {
	gdf_time_unit time_unit;
	// here we can also hold info for decimal datatype or any other datatype that requires additional information
} gdf_dtype_extra_info;

typedef struct gdf_column_{
    void *data;                       /**< Pointer to the columns data */
    gdf_valid_type *valid;            /**< Pointer to the columns validity bit mask where the 'i'th bit indicates if the 'i'th row is NULL */
    gdf_size_type size;               /**< Number of data elements in the columns data buffer. Limited to 2^31 - 1.*/
    gdf_dtype dtype;                  /**< The datatype of the column's data */
    gdf_size_type null_count;         /**< The number of NULL values in the column's data */
    gdf_dtype_extra_info dtype_info;
    char *			col_name;			// host-side:	null terminated string
} gdf_column;

struct CsrCudf {
  Entry* data;
  size_t* offsets;
  size_t n_nz;
  size_t n_rows;
  size_t n_cols;
};

void RunConverter(gdf_column** gdf_data, CsrCudf* csr);

//--- private CUDA functions / kernels
__global__ void cuda_create_csr_k
(void *cudf_data, gdf_valid_type* valid, gdf_dtype dtype, int col, Entry* data,
 gdf_size_type *offsets, size_t n_rows);

__global__ void determine_valid_rec_count_k
(gdf_valid_type* valid, size_t n_rows, size_t n_cols, size_t* offset);

__device__ int WhichBitmap(int record) { return record / 8; }
__device__ int WhichBit(int bit) { return bit % 8; }
__device__ int CheckBit(gdf_valid_type data, int bit) {
  gdf_valid_type bit_mask[8] = {1, 2, 4, 8, 16, 32, 64, 128};
  return data & bit_mask[bit];
}

__device__ bool IsValid(gdf_valid_type* valid, int tid) {
  if (valid == nullptr)
    return true;
  int bitmap_idx = WhichBitmap(tid);
  int bit_idx = WhichBit(tid);
  gdf_valid_type bitmap = valid[bitmap_idx];
  return CheckBit(bitmap, bit_idx);
}

// Convert a CUDF into a CSR CUDF
void CUDFToCSR(gdf_column** cudf_data, int n_cols, CsrCudf* csr) {
  size_t n_rows = cudf_data[0]->size;

  // the first step is to create an array that counts the number of valid entries per row
  // this is done by each thread looking across its row and checking the valid bits
  int threads = 1024;
  int blocks = (n_rows + threads - 1) / threads;

  size_t* offsets = csr->offsets;
  dh::safe_cuda(cudaMemset(offsets, 0, sizeof(gdf_size_type) * (n_rows + 1)));

  if (blocks > 0) {
    for (int i = 0; i < n_cols; ++i) {
      determine_valid_rec_count_k<<<blocks, threads>>>
        (cudf_data[i]->valid, n_rows, n_cols, offsets);
      dh::safe_cuda(cudaGetLastError());
      dh::safe_cuda(cudaDeviceSynchronize());
    }
  }

  // compute the number of elements
  thrust::device_ptr<size_t> offsets_begin(offsets);
  int64_t n_elements = thrust::reduce
    (offsets_begin, offsets_begin + n_rows, 0ull, thrust::plus<size_t>());

  // now do an exclusive scan to compute the offsets for where to write data
  thrust::exclusive_scan(offsets_begin, offsets_begin + n_rows + 1, offsets_begin);

  csr->n_rows = n_rows;
  csr->n_cols = n_cols;
  csr->n_nz = n_elements;

  // process based on data type
  RunConverter(cudf_data, csr);
}

void RunConverter(gdf_column** cudf_data, CsrCudf* csr) {
  size_t n_cols = csr->n_cols;
  size_t n_rows = csr->n_rows;

  int threads = 256;
  int blocks = (n_rows + threads - 1) / threads;

  // temporary offsets for writing data
  thrust::device_ptr<size_t> offset_begin(csr->offsets);
  thrust::device_vector<size_t> offsets2(offset_begin, offset_begin + n_rows + 1);

  // move the data and create the CSR
  if (blocks > 0) {
    for (int col = 0; col < n_cols; ++col) {
      gdf_column *cudf = cudf_data[col];
      cuda_create_csr_k<<<blocks, threads>>>
        (cudf->data, cudf->valid, cudf->dtype, col, csr->data,
         offsets2.data().get(), n_rows);
      dh::safe_cuda(cudaGetLastError());
    }
  }
}

// move data over into CSR and possibly convert the format
__global__ void cuda_create_csr_k
(void* cudf_data, gdf_valid_type* valid, gdf_dtype dtype, int col,
 Entry* data, size_t* offsets, size_t n_rows) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= n_rows)
    return;
  gdf_size_type offset_idx = offsets[tid];
  if (IsValid(valid, tid)) {
    data[offset_idx].fvalue = ConvertDataElement(cudf_data, tid, dtype);
    data[offset_idx].index = col;
    ++offsets[tid];
  }
}

// compute the number of valid entries per row
__global__ void determine_valid_rec_count_k
(gdf_valid_type *valid, size_t n_rows, size_t n_cols, gdf_size_type* offset) {

  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= n_rows)
    return;
  if (IsValid(valid, tid))
    ++offset[tid];
}

void InitFromCUDF(SimpleCSRSource *source, gdf_column** cols, size_t n_cols) {
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
  CsrCudf csr;
  csr.data = page_.data.DevicePointer(0);
  csr.offsets = page_.offset.DevicePointer(0);
  csr.n_nz = 0;
  csr.n_rows = n_rows;
  csr.n_cols = n_cols;
  CUDFToCSR(cols, n_cols, &csr);
}

int XGDMatrixCreateFromCUDF
(void **cols, size_t n_cols, DMatrixHandle *out) {
  API_BEGIN();
  std::unique_ptr<data::SimpleCSRSource> source(new data::SimpleCSRSource());
  InitFromCUDF(source.get(), cols, n_cols);
  *out = new std::shared_ptr<DMatrix>(DMatrix::Create(std::move(source)));
  API_END();
}


XGB_DLL int XGDMatrixSetCUDFInfo(DMatrixHandle handle,
                                const char *field,
                                void **cols,
                                size_t n_cols) {
  API_BEGIN();
  CHECK_HANDLE();
  // static_cast<std::shared_ptr<DMatrix>*>(handle)
  //  ->get()->Info().SetCUDFInfo(field, cols, n_cols);
  API_END();
}
}

