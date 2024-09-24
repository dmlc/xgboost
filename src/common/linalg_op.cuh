/**
 * Copyright 2021-2023, XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_LINALG_OP_CUH_
#define XGBOOST_COMMON_LINALG_OP_CUH_

#include <cstdint>  // for int32_t
#include <cstdlib>  // for size_t
#include <tuple>    // for apply

#include "device_helpers.cuh"  // for LaunchN
#include "linalg_op.h"
#include "xgboost/context.h"  // for Context
#include "xgboost/linalg.h"   // for TensorView

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
    dh::LaunchN(t.Size(), s, [=] __device__(std::size_t i) { fn(i); });
  }
};

template <typename T, std::int32_t D, typename Fn>
void ElementWiseKernel(TensorView<T, D> t, Fn&& fn, cudaStream_t s = nullptr) {
  dh::safe_cuda(cudaSetDevice(t.Device().ordinal));
  cuda_impl::ElementWiseImpl<T, D>{}(t, fn, s);
}
}  // namespace cuda_impl

template <typename T, int32_t D, typename Fn>
void ElementWiseTransformDevice(TensorView<T, D> t, Fn&& fn, cudaStream_t s = nullptr) {
  if (t.Contiguous()) {
    auto ptr = t.Values().data();
    dh::LaunchN(t.Size(), s, [=] __device__(size_t i) { ptr[i] = fn(i, ptr[i]); });
  } else {
    dh::LaunchN(t.Size(), s, [=] __device__(size_t i) mutable {
      T& v = detail::Apply(t, UnravelIndex(i, t.Shape()));
      v = fn(i, v);
    });
  }
}

template <typename T, int32_t D, typename Fn>
void ElementWiseKernel(Context const* ctx, TensorView<T, D> t, Fn&& fn) {
  ctx->IsCUDA() ? cuda_impl::ElementWiseKernel(t, fn)
                : ElementWiseKernelHost(t, ctx->Threads(), fn);
}

namespace detail {
template <typename T, std::int32_t kDim>
struct IterOp {
  TensorView<T, kDim> v;
  XGBOOST_DEVICE T& operator()(std::size_t i) {
    return detail::Apply(v, UnravelIndex(i, v.Shape()));
  }
};
}  // namespace detail

// naming: thrust begin
// returns a thrust iterator for a tensor view.
template <typename T, std::int32_t kDim>
auto tcbegin(TensorView<T, kDim> v) {  // NOLINT
  return thrust::make_transform_iterator(
      thrust::make_counting_iterator(0ul),
      detail::IterOp<std::add_const_t<std::remove_const_t<T>>, kDim>{v});
}

template <typename T, std::int32_t kDim>
auto tcend(TensorView<T, kDim> v) {  // NOLINT
  return tcbegin(v) + v.Size();
}

template <typename T, std::int32_t kDim>
auto tbegin(TensorView<T, kDim> v) {  // NOLINT
  return thrust::make_transform_iterator(thrust::make_counting_iterator(0ul),
                                         detail::IterOp<std::remove_const_t<T>, kDim>{v});
}

template <typename T, std::int32_t kDim>
auto tend(TensorView<T, kDim> v) {  // NOLINT
  return tbegin(v) + v.Size();
}
}  // namespace xgboost::linalg
#endif  // XGBOOST_COMMON_LINALG_OP_CUH_
