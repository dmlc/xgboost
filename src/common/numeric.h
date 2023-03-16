/**
 * Copyright 2022-2023 by XGBoost contributors.
 */
#ifndef XGBOOST_COMMON_NUMERIC_H_
#define XGBOOST_COMMON_NUMERIC_H_

#include <dmlc/common.h>  // OMPException

#include <algorithm>  // for std::max
#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t
#include <iterator>   // for iterator_traits
#include <vector>

#include "common.h"                      // AssertGPUSupport
#include "threading_utils.h"             // MemStackAllocator, DefaultMaxThreads
#include "xgboost/context.h"             // Context
#include "xgboost/host_device_vector.h"  // HostDeviceVector

namespace xgboost::common {

/**
 * \brief Run length encode on CPU, input must be sorted.
 */
template <typename Iter, typename Idx>
void RunLengthEncode(Iter begin, Iter end, std::vector<Idx>* p_out) {
  auto& out = *p_out;
  out = std::vector<Idx>{0};
  size_t n = std::distance(begin, end);
  for (size_t i = 1; i < n; ++i) {
    if (begin[i] != begin[i - 1]) {
      out.push_back(i);
    }
  }
  if (out.back() != n) {
    out.push_back(n);
  }
}

/**
 * \brief Varient of std::partial_sum, out_it should point to a container that has n + 1
 *        elements. Useful for constructing a CSR indptr.
 */
template <typename InIt, typename OutIt, typename T>
void PartialSum(int32_t n_threads, InIt begin, InIt end, T init, OutIt out_it) {
  static_assert(std::is_same<T, typename std::iterator_traits<InIt>::value_type>::value);
  static_assert(std::is_same<T, typename std::iterator_traits<OutIt>::value_type>::value);
  // The number of threads is pegged to the batch size. If the OMP block is parallelized
  // on anything other than the batch/block size, it should be reassigned
  auto n = static_cast<size_t>(std::distance(begin, end));
  const size_t batch_threads =
      std::max(static_cast<size_t>(1), std::min(n, static_cast<size_t>(n_threads)));
  MemStackAllocator<T, DefaultMaxThreads()> partial_sums(batch_threads);

  size_t block_size = n / batch_threads;

  dmlc::OMPException exc;
#pragma omp parallel num_threads(batch_threads)
  {
#pragma omp for
    for (omp_ulong tid = 0; tid < batch_threads; ++tid) {
      exc.Run([&]() {
        size_t ibegin = block_size * tid;
        size_t iend = (tid == (batch_threads - 1) ? n : (block_size * (tid + 1)));

        T running_sum = 0;
        for (size_t ridx = ibegin; ridx < iend; ++ridx) {
          running_sum += *(begin + ridx);
          *(out_it + 1 + ridx) = running_sum;
        }
      });
    }

#pragma omp single
    {
      exc.Run([&]() {
        partial_sums[0] = init;
        for (size_t i = 1; i < batch_threads; ++i) {
          partial_sums[i] = partial_sums[i - 1] + *(out_it + i * block_size);
        }
      });
    }

#pragma omp for
    for (omp_ulong tid = 0; tid < batch_threads; ++tid) {
      exc.Run([&]() {
        size_t ibegin = block_size * tid;
        size_t iend = (tid == (batch_threads - 1) ? n : (block_size * (tid + 1)));

        for (size_t i = ibegin; i < iend; ++i) {
          *(out_it + 1 + i) += partial_sums[tid];
        }
      });
    }
  }
  exc.Rethrow();
}

namespace cuda_impl {
double Reduce(Context const* ctx, HostDeviceVector<float> const& values);
#if !defined(XGBOOST_USE_CUDA)
inline double Reduce(Context const*, HostDeviceVector<float> const&) {
  AssertGPUSupport();
  return 0;
}
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace cuda_impl

/**
 * \brief Reduction with iterator. init must be additive identity. (0 for primitive types)
 */
namespace cpu_impl {
template <typename It, typename V = typename It::value_type>
V Reduce(Context const* ctx, It first, It second, V const& init) {
  std::size_t n = std::distance(first, second);
  auto n_threads = static_cast<std::size_t>(std::min(n, static_cast<std::size_t>(ctx->Threads())));
  common::MemStackAllocator<V, common::DefaultMaxThreads()> result_tloc(n_threads, init);
  common::ParallelFor(n, n_threads, [&](auto i) { result_tloc[omp_get_thread_num()] += first[i]; });
  auto result = std::accumulate(result_tloc.cbegin(), result_tloc.cbegin() + n_threads, init);
  return result;
}
}  // namespace cpu_impl

/**
 * \brief Reduction on host device vector.
 */
double Reduce(Context const* ctx, HostDeviceVector<float> const& values);

template <typename It>
void Iota(Context const* ctx, It first, It last,
          typename std::iterator_traits<It>::value_type const& value) {
  auto n = std::distance(first, last);
  std::int32_t n_threads = ctx->Threads();
  const size_t block_size = n / n_threads + !!(n % n_threads);
  dmlc::OMPException exc;
#pragma omp parallel num_threads(n_threads)
  {
    exc.Run([&]() {
      const size_t tid = omp_get_thread_num();
      const size_t ibegin = tid * block_size;
      const size_t iend = std::min(ibegin + block_size, static_cast<size_t>(n));
      for (size_t i = ibegin; i < iend; ++i) {
        first[i] = i + value;
      }
    });
  }
}
}  // namespace xgboost::common

#endif  // XGBOOST_COMMON_NUMERIC_H_
