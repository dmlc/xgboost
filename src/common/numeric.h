/*!
 * Copyright 2022, XGBoost contributors.
 */
#ifndef XGBOOST_COMMON_NUMERIC_H_
#define XGBOOST_COMMON_NUMERIC_H_

#include <algorithm>  // std::max
#include <iterator>   // std::iterator_traits
#include <vector>

#include "threading_utils.h"             // MemStackAllocator, DefaultMaxThreads
#include "xgboost/generic_parameters.h"  // Context
#include "xgboost/host_device_vector.h"  // HostDeviceVector

namespace xgboost {
namespace common {

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
  static_assert(std::is_same<T, typename std::iterator_traits<InIt>::value_type>::value, "");
  static_assert(std::is_same<T, typename std::iterator_traits<OutIt>::value_type>::value, "");
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

namespace cuda {
double Reduce(Context const* ctx, HostDeviceVector<float> const& values);
}
/**
 * \brief Reduction with summation.
 */
double Reduce(Context const* ctx, HostDeviceVector<float> const& values);
}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_NUMERIC_H_
