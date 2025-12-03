/**
 * Copyright 2021-2025, XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_LINALG_OP_CUH_
#define XGBOOST_COMMON_LINALG_OP_CUH_

#include <thrust/iterator/counting_iterator.h>  // for counting_iterator
#include <thrust/iterator/zip_iterator.h>       // for make_zip_iterator
#include <thrust/transform.h>                   // for transform

#include <cstdint>            // for int32_t
#include <cstdlib>            // for size_t
#include <cuda/functional>    // for proclaim_copyable_arguments
#include <cuda/std/iterator>  // for iterator_traits
#include <cuda/std/tuple>     // for get
#include <tuple>              // for apply

#include "cuda_context.cuh"
#include "device_helpers.cuh"  // for LaunchN
#include "type.h"              // for GetValueT
#include "xgboost/context.h"   // for Context
#include "xgboost/linalg.h"    // for TensorView

namespace xgboost::linalg {
namespace cuda_impl {
// Use template specialization to dispatch, Windows + CUDA 11.8 doesn't support extended
// lambda inside constexpr if
template <typename T, std::int32_t D>
struct ElementWiseImpl {
  template <typename Fn>
  void operator()(TensorView<T, D> t, Fn&& fn, cudaStream_t s) {
    static_assert(D > 1);
    dh::LaunchN(t.Size(), s, [=] __device__(std::size_t i) mutable {
      std::apply(fn, linalg::UnravelIndex(i, t.Shape()));
    });
  }
};

template <typename T>
struct ElementWiseImpl<T, 1> {
  template <typename Fn>
  void operator()(TensorView<T, 1> t, Fn&& fn, cudaStream_t s) {
    dh::LaunchN(t.Size(), s, [=] __device__(std::size_t i) mutable { fn(i); });
  }
};

template <typename T, std::int32_t D, typename Fn>
void ElementWiseKernel(TensorView<T, D> t, Fn&& fn, cudaStream_t s = nullptr) {
  dh::safe_cuda(cudaSetDevice(t.Device().ordinal));
  ElementWiseImpl<T, D>{}(t, fn, s);
}

template <typename T, std::int32_t D, typename Fn>
void TransformIdxKernel(Context const* ctx, TensorView<T, D> t, Fn&& fn) {
  dh::safe_cuda(cudaSetDevice(t.Device().ordinal));
  auto s = ctx->CUDACtx()->Stream();
  if (t.Contiguous()) {
    auto ptr = t.Values().data();
    auto it =
        thrust::make_zip_iterator(thrust::make_counting_iterator(static_cast<std::size_t>(0)), ptr);
    using Tuple = typename cuda::std::iterator_traits<common::GetValueT<decltype(it)>>::value_type;
    thrust::transform(ctx->CUDACtx()->CTP(), it, it + t.Size(), ptr,
                      [=] XGBOOST_DEVICE(Tuple const& tup) {
                        return fn(cuda::std::get<0>(tup), cuda::std::get<1>(tup));
                      });
  } else {
    dh::LaunchN(t.Size(), s, [=] __device__(size_t i) mutable {
      T& v = std::apply(t, UnravelIndex(i, t.Shape()));
      v = fn(i, v);
    });
  }
}

template <typename T, std::int32_t D, typename Fn>
void TransformKernel(Context const* ctx, TensorView<T, D> t, Fn&& fn) {
  dh::safe_cuda(cudaSetDevice(t.Device().ordinal));
  auto s = ctx->CUDACtx()->Stream();
  if (t.Contiguous()) {
    auto ptr = t.Values().data();
    thrust::transform(
        ctx->CUDACtx()->CTP(), ptr, ptr + t.Size(), ptr,
        cuda::proclaim_copyable_arguments([=] XGBOOST_DEVICE(T const& v) { return fn(v); }));
  } else {
    dh::LaunchN(t.Size(), s, [=] __device__(size_t i) mutable {
      T& v = std::apply(t, UnravelIndex(i, t.Shape()));
      v = fn(v);
    });
  }
}
}  // namespace cuda_impl

namespace detail {
template <typename T, std::int32_t D>
struct IterOp {
  TensorView<T, D> v;
  XGBOOST_DEVICE T& operator()(std::size_t i) { return std::apply(v, UnravelIndex(i, v.Shape())); }
};
}  // namespace detail

// naming: thrust begin
// returns a thrust iterator for a tensor view.
template <typename T, std::int32_t D>
auto tcbegin(TensorView<T, D> v) {  // NOLINT
  return thrust::make_transform_iterator(
      thrust::make_counting_iterator(0ul),
      detail::IterOp<std::add_const_t<std::remove_const_t<T>>, D>{v});
}

template <typename T, std::int32_t D>
auto tcend(TensorView<T, D> v) {  // NOLINT
  return tcbegin(v) + v.Size();
}

template <typename T, std::int32_t D>
auto tbegin(TensorView<T, D> v) {  // NOLINT
  return thrust::make_transform_iterator(thrust::make_counting_iterator(0ul),
                                         detail::IterOp<std::remove_const_t<T>, D>{v});
}

template <typename T, std::int32_t D>
auto tend(TensorView<T, D> v) {  // NOLINT
  return tbegin(v) + v.Size();
}
}  // namespace xgboost::linalg
#endif  // XGBOOST_COMMON_LINALG_OP_CUH_
