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

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

#include "xgboost4j_spark_gpu.h"

namespace xgboost {
namespace spark {

// This is from the strided_range thrust example.
template <typename iter_type>
class strided_range {
public:
  typedef typename thrust::iterator_difference<iter_type>::type difference_type;

  struct stride_functor : public thrust::unary_function<difference_type,difference_type> {
    difference_type stride;

    stride_functor(difference_type stride) : stride(stride) {}

    __host__ __device__
    difference_type operator()(const difference_type& i) const {
      return stride * i;
    }
  };

  typedef typename thrust::counting_iterator<difference_type>
      count_iterator;
  typedef typename thrust::transform_iterator<stride_functor, count_iterator>
      transform_iterator;
  typedef typename thrust::permutation_iterator<iter_type, transform_iterator>
      permute_iterator;

  // type of the strided_range iterator
  typedef permute_iterator iterator;

  // construct strided_range for the range [first,last)
  strided_range(iter_type first, iter_type last, difference_type stride)
    : first(first), last(last), stride(stride) {}

  iterator begin(void) const {
    return permute_iterator(first,
        transform_iterator(count_iterator(0), stride_functor(stride)));
  }

  iterator end(void) const {
    return begin() + ((last - first) + (stride - 1)) / stride;
  }

protected:
  iter_type first;
  iter_type last;
  difference_type stride;
};

template <typename store_type>
void stride_store(void* dest, store_type const* src, long count,
                  int32_t byte_stride, cudaStream_t stream = 0) {
  store_type* o_data = static_cast<store_type*>(dest);
  // Issues with int division are handled at a higher level
  int32_t typed_stride = byte_stride / sizeof(store_type);

  auto input_dptr = thrust::device_pointer_cast(src);
  strided_range<store_type*> iter(o_data, o_data + typed_stride * count,
                                  typed_stride);
  thrust::copy(thrust::device, input_dptr, input_dptr + count, iter.begin());
}

cudaError_t store_with_stride_async(void* dest, void const* src, long count,
                                    int byte_width, int byte_stride,
                                    cudaStream_t stream) {
  switch (byte_width) {
  case 1:
    stride_store<int8_t>(dest, static_cast<int8_t const*>(src), count,
                         byte_stride, stream);
    break;
  case 2:
    stride_store<int16_t>(dest, static_cast<int16_t const*>(src), count,
                          byte_stride, stream);
    break;
  case 4:
    stride_store<int32_t>(dest, static_cast<int32_t const*>(src), count,
                          byte_stride, stream);
    break;
  case 8:
    stride_store<int64_t>(dest, static_cast<int64_t const*>(src), count,
                          byte_stride, stream);
    break;
  default:
    return cudaErrorInvalidValue;
  }

  return cudaGetLastError();
}

cudaError_t build_unsaferow_nullsets(uint64_t* dest,
                                     const uint32_t* const* validity_vectors,
                                     int num_vectors, unsigned int rows) {
  auto dest_dptr = thrust::device_pointer_cast(dest);
  const int num_nullset_longs_per_row = (num_vectors + 63) / 64;
  const int num_longs_per_row = num_nullset_longs_per_row + num_vectors;
  const int num_trailing_bits = num_vectors % 64;

  // Convert the index relative to the virtual address space of all
  // row null bitsets concatenated together into a long index relative
  // to the UnsafeRow data.
  auto index_transformer = [=] __device__(ptrdiff_t idx) {
    int row = idx / num_nullset_longs_per_row;
    int nullset_long_idx = idx % num_nullset_longs_per_row;
    return row * num_longs_per_row + nullset_long_idx;
  };
  auto xform_iter = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0), index_transformer);
  auto permute_iter = thrust::make_permutation_iterator(dest_dptr, xform_iter);

  // Build a transformer that will perform the cudf column validity vector to
  // Spark UnsafeRow null bitset conversion. The index parameter passed in
  // is relative to all of the 64-bit words of the row null bitsets as if they
  // were in a contiguous address space.
  auto nullset_builder = [=] __device__(size_t iter_idx) {
    int row = iter_idx / num_nullset_longs_per_row;
    int nullset_long_idx = iter_idx % num_nullset_longs_per_row;
    int num_bits_per_row = 64;
    if (nullset_long_idx == num_nullset_longs_per_row - 1) {
      num_bits_per_row = num_trailing_bits;
    }

    // Get a base pointer to the array of (up to 64) validity vectors that
    // will be accessed to fill this long word of the row's null bitset.
    const uint32_t* const* vvecs = validity_vectors + nullset_long_idx * 64;

    int vvec_bitshift = row % 32;
    int vvec_byte_idx = row / 32;
    uint64_t null_mask = 0;
    for (int i = 0; i < num_bits_per_row; ++i) {
      if (vvecs[i] != NULL) {
        uint32_t valid_byte = vvecs[i][vvec_byte_idx];
        uint64_t nullbit = (~valid_byte >> vvec_bitshift) & 0x1;
        null_mask |= nullbit << i;
      }
    }

    return null_mask;
  };

  thrust::tabulate(permute_iter, permute_iter + num_nullset_longs_per_row * rows,
                   nullset_builder);

  return cudaGetLastError();
}

} // namespace spark
} // namespace xgboost
