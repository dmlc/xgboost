/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <vector>
#include <memory>
#include <iostream>

#include <cuda_runtime.h>

#include "xgboost4j_spark_gpu.h"
#include "xgboost4j_spark.h"

#if defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1
#include "rmm/mr/device/per_device_resource.hpp"
#include "rmm/mr/device/thrust_allocator_adaptor.hpp"
#include "rmm/device_buffer.hpp"
#endif  // defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1

struct gpu_column_data {
  long* data_ptr;
  long* valid_ptr;
  int dtype_size_in_bytes;
  long num_row;
};

namespace xgboost {
namespace spark {

/*! \brief utility class to track GPU allocations */
class unique_gpu_ptr {
  void* ptr;
  const size_t size;

public:
  unique_gpu_ptr(unique_gpu_ptr const&) = delete;
  unique_gpu_ptr& operator=(unique_gpu_ptr const&) = delete;

  unique_gpu_ptr(unique_gpu_ptr&& other) noexcept : ptr(other.ptr), size(other.size) {
    other.ptr = nullptr;
  }

  unique_gpu_ptr(size_t _size) : ptr(nullptr), size(_size) {
#if defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1
    rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource();
    try {
      ptr = mr->allocate(_size);
    } catch (rmm::bad_alloc const& e) {
      auto what = std::string("Could not allocate memory from RMM: ") +
        (e.what() == nullptr ? "" : e.what());
      std::cerr << what << std::endl;
      throw std::bad_alloc();
    }
#else
    cudaError_t status = cudaMalloc(&ptr, _size);
    if (status != cudaSuccess) {
      throw std::bad_alloc();
    }
#endif
  }

  ~unique_gpu_ptr() {
    if (ptr != nullptr) {
#if defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1
      rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource();
      mr->deallocate(ptr, size);
#else
      cudaFree(ptr);
#endif
    }
  }

  void* get() {
    return ptr;
  }

  void* release() {
    void* result = ptr;
    ptr = nullptr;
    return result;
  }
};

/*! \brief custom deleter to free malloc allocations */
struct malloc_deleter {
  void operator()(void* ptr) const {
    free(ptr);
  }
};

static unsigned int get_unsaferow_nullset_size(unsigned int num_columns) {
  // The nullset size is rounded up to a multiple of 8 bytes.
  return ((num_columns + 63) / 64) * 8;
}

static void build_unsafe_row_nullsets(void* unsafe_rows_dptr,
                                      std::vector<gpu_column_data *> const& gdfcols) {
  unsigned int num_columns = gdfcols.size();
  size_t num_rows = gdfcols[0]->num_row;

  // make the array of validity data pointers available on the device
  std::vector<uint32_t const*> valid_ptrs(num_columns);
  for (int i = 0; i < num_columns; ++i) {
    valid_ptrs[i] = reinterpret_cast<const unsigned int *>(gdfcols[i]->valid_ptr);
  }
  unique_gpu_ptr dev_valid_mem(num_columns * sizeof(*valid_ptrs.data()));
  uint32_t** dev_valid_ptrs = reinterpret_cast<uint32_t**>(dev_valid_mem.get());
  cudaError_t cuda_status = cudaMemcpy(dev_valid_ptrs, valid_ptrs.data(),
                                       num_columns * sizeof(valid_ptrs[0]), cudaMemcpyHostToDevice);
  if (cuda_status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(cuda_status));
  }

  // build the nullsets for each UnsafeRow
  cuda_status = xgboost::spark::build_unsaferow_nullsets(
      reinterpret_cast<uint64_t*>(unsafe_rows_dptr), dev_valid_ptrs,
      num_columns, num_rows);
  if (cuda_status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(cuda_status));
  }
}


/*!
 * \brief Transforms a set of cudf columns into an array of Spark UnsafeRow.
 * NOTE: Only fixed-length datatypes are supported, as it is assumed
 * that every UnsafeRow has the same size.
 *
 * Spark's UnsafeRow with fixed-length datatypes has the following format:
 *   null bitset, 8-byte value, [8-byte value, ...]
 * where the null bitset is a collection of 64-bit words with each bit
 * indicating whether the corresponding field is null.
 */
void* build_unsafe_rows(std::vector<gpu_column_data *> const& gdfcols) {
  cudaError_t cuda_status;
  unsigned int num_columns = gdfcols.size();
  size_t num_rows = gdfcols[0]->num_row;
  unsigned int nullset_size = get_unsaferow_nullset_size(num_columns);
  unsigned int row_size = nullset_size + num_columns * 8;
  size_t unsafe_rows_size = num_rows * row_size;

  // allocate GPU memory to hold the resulting UnsafeRow array
  unique_gpu_ptr unsafe_rows_devmem(unsafe_rows_size);
  uint8_t* unsafe_rows_dptr = static_cast<uint8_t*>(unsafe_rows_devmem.get());

  // write each column to the corresponding position in the unsafe rows
  for (int i = 0; i < num_columns; ++i) {
    // point to the corresponding field in the first UnsafeRow
    uint8_t* dest_addr = unsafe_rows_dptr + nullset_size + i * 8;
    int dtype_size = gdfcols[i]->dtype_size_in_bytes;
    if (dtype_size <= 0) {
      throw std::runtime_error("Unsupported column type");
    }
    cuda_status = xgboost::spark::store_with_stride_async(dest_addr,
        gdfcols[i]->data_ptr, num_rows, dtype_size, row_size, 0);
    if (cuda_status != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(cuda_status));
    }
  }

  build_unsafe_row_nullsets(unsafe_rows_dptr, gdfcols);

  // copy UnsafeRow results back to host
  std::unique_ptr<void, malloc_deleter> unsafe_rows(malloc(unsafe_rows_size));
  if (unsafe_rows.get() == nullptr) {
    throw std::bad_alloc();
  }
  // This copy also serves as a synchronization point with the GPU.
  cuda_status = cudaMemcpy(unsafe_rows.get(), unsafe_rows_dptr,
      unsafe_rows_size, cudaMemcpyDeviceToHost);
  if (cuda_status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(cuda_status));
  }

  return unsafe_rows.release();
}
} // namespace spark
} // namespace xgboost


static void throw_java_exception(JNIEnv* env, char const* classname,
    char const* msg) {
  jclass exClass = env->FindClass(classname);
  if (exClass != NULL) {
    env->ThrowNew(exClass, msg);
  }
}

static void throw_java_exception(JNIEnv* env, char const* msg) {
  throw_java_exception(env, "java/lang/RuntimeException", msg);
}

JNIEXPORT jlong JNICALL
Java_ml_dmlc_xgboost4j_java_XGBoostSparkJNI_buildUnsafeRows(JNIEnv * env,
    jclass clazz, jlongArray dataPtrs, jlongArray validPtrs, jintArray dTypePtrs, jlong numRows) {
  // FIXME, Do we need to check if the size of dataPtrs/validPtrs/dTypePtrs should be same
  int num_columns = env->GetArrayLength(dataPtrs);

  if (env->ExceptionOccurred()) {
    return 0;
  }
  if (num_columns <= 0) {
    throw_java_exception(env, "Invalid number of columns");
    return 0;
  }

  std::vector<gpu_column_data *> gpu_cols;

  jlong* data_jlongs = env->GetLongArrayElements(dataPtrs, nullptr);
  if (data_jlongs == nullptr) {
    throw_java_exception(env, "Failed to get data handles");
    return 0;
  }
  jlong* valid_jlongs = env->GetLongArrayElements(validPtrs, nullptr);
  if (valid_jlongs == nullptr) {
    throw_java_exception(env, "Failed to get valid handles");
    return 0;
  }
  jint* dtype_jints = env->GetIntArrayElements(dTypePtrs, nullptr);
  if (dtype_jints == nullptr) {
    throw_java_exception(env, "Failed to get data type sizes");
    return 0;
  }
  for (int i = 0; i < num_columns; ++i) {
    gpu_column_data* tmp_column = new gpu_column_data;
    tmp_column->data_ptr = reinterpret_cast<long * > (data_jlongs[i]);
    tmp_column->valid_ptr = reinterpret_cast<long * > (valid_jlongs[i]);
    tmp_column->dtype_size_in_bytes = dtype_jints[i];
    tmp_column->num_row = numRows;
    gpu_cols.push_back(tmp_column);
  }

  env->ReleaseLongArrayElements(dataPtrs, data_jlongs, JNI_ABORT);
  env->ReleaseLongArrayElements(validPtrs, valid_jlongs, JNI_ABORT);
  env->ReleaseIntArrayElements(dTypePtrs, dtype_jints, JNI_ABORT);

  void* unsafe_rows = nullptr;
  try {
    unsafe_rows = xgboost::spark::build_unsafe_rows(gpu_cols);
  } catch (std::bad_alloc const& e) {
    throw_java_exception(env, "java/lang/OutOfMemoryError",
                         "Could not allocate native memory");
  } catch (std::exception const& e) {
    throw_java_exception(env, e.what());
  }
  for (int i = 0; i < num_columns; i++) {
    delete (gpu_cols[i]);
  }
  gpu_cols.clear();

  return reinterpret_cast<jlong>(unsafe_rows);
}

JNIEXPORT jint JNICALL
Java_ml_dmlc_xgboost4j_java_XGBoostSparkJNI_getGpuDevice(JNIEnv * env,
    jclass clazz) {
    int device_ordinal;
    cudaGetDevice(&device_ordinal);
    return device_ordinal;
}

JNIEXPORT jint JNICALL
Java_ml_dmlc_xgboost4j_java_XGBoostSparkJNI_allocateGpuDevice(JNIEnv * env,
    jclass clazz, jint gpu_id) {

    cudaError_t error = cudaSetDevice(gpu_id);
    if (error != cudaSuccess) {
       throw_java_exception(env, "Error running cudaSetDevice");
    }
    // initialize a context
    error = cudaFree(0);
    if (error != cudaSuccess) {
      throw_java_exception(env, "Error running cudaFree");
    }
    return 0;
}
