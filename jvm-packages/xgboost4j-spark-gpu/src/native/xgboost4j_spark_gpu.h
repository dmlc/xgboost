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

#ifndef INCLUDED_XGBOOST4J_SPARK_GPU_H
#define INCLUDED_XGBOOST4J_SPARK_GPU_H

#include <cstdint>
#include <cuda_runtime.h>

namespace xgboost {
namespace spark {

cudaError_t store_with_stride_async(void* dest, void const* src, long count,
                                    int byte_width, int byte_stride,
                                    cudaStream_t stream);

cudaError_t build_unsaferow_nullsets(uint64_t* dest,
                                     const uint32_t* const* validity_vectors,
                                     int num_vectors, unsigned int rows);

} // namespace spark
} // namespace xgboost

#endif // INCLUDED_XGBOOST4J_SPARK_GPU_H
