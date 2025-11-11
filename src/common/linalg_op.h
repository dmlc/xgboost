/**
 * Copyright 2021-2025, XGBoost Contributors
 *
 * @brief This module defines the dispatching functions for various linalg kernels.
 *
 * Client code can use utilities like @ref ElementWiseKernel by including this file in the
 * right translation unit. For CUDA-compatible kernels, include this header in a .cu TU.
 */
#ifndef XGBOOST_COMMON_LINALG_OP_H_
#define XGBOOST_COMMON_LINALG_OP_H_

#include <cstddef>      // for size_t
#include <cstdint>      // for int32_t
#include <tuple>        // for apply
#include <type_traits>  // for conditional_t

#include "json_utils.h"  // for LoadVector, SaveVector
#include "threading_utils.h"
#include "transform_iterator.h"  // for MakeIndexTransformIter
#include "xgboost/json.h"        // for Json
#include "xgboost/linalg.h"

#if defined(__CUDACC__)
#include <utility>  // for forward

#include "linalg_op.cuh"
#endif

#if defined(XGBOOST_USE_SYCL)
#include "../../plugin/sycl/common/linalg_op.h"
#endif

#if !defined(XGBOOST_USE_CUDA) && !defined(XGBOOST_USE_SYCL)

#include "common.h"           // for AssertGPUSupport
#include "xgboost/context.h"  // for Context

#endif  // !defined(XGBOOST_USE_CUDA) && !defined(XGBOOST_USE_SYCL)

namespace xgboost::common {
struct OptionalWeights;
}

namespace xgboost::linalg {
namespace cpu_impl {
template <typename T, std::int32_t D, typename Fn>
void TransformIdxKernel(linalg::TensorView<T, D> t, std::int32_t n_threads, Fn&& fn) {
  if (t.Contiguous()) {
    auto ptr = t.Values().data();
    common::ParallelFor(t.Size(), n_threads, [&](std::size_t i) { ptr[i] = fn(i, ptr[i]); });
  } else {
    common::ParallelFor(t.Size(), n_threads, [&](std::size_t i) {
      auto& v = std::apply(t, linalg::UnravelIndex(i, t.Shape()));
      v = fn(i, v);
    });
  }
}

template <typename T, std::int32_t D, typename Fn>
void TransformKernel(linalg::TensorView<T, D> t, std::int32_t n_threads, Fn&& fn) {
  if (t.Contiguous()) {
    auto ptr = t.Values().data();
    common::ParallelFor(t.Size(), n_threads, [&](std::size_t i) { ptr[i] = fn(ptr[i]); });
  } else {
    common::ParallelFor(t.Size(), n_threads, [&](std::size_t i) {
      auto& v = std::apply(t, linalg::UnravelIndex(i, t.Shape()));
      v = fn(v);
    });
  }
}

template <typename T, std::int32_t D, typename Fn>
void ElementWiseKernel(linalg::TensorView<T, D> t, std::int32_t n_threads, Fn&& fn) {
  constexpr std::size_t kBlockSize = 2048;
  if constexpr (D == 1) {
    common::ParallelFor1d<kBlockSize>(t.Size(), n_threads, [&](auto&& block) {
      for (std::size_t i = block.begin(); i < block.end(); ++i) {
        fn(i);
      }
    });
  } else if (D == 2 && t.CContiguous() && t.Shape(0) > t.Shape(1) * 64) {
    // Heuristic. Tall, c-contiguous matrix,
    auto n_rows = t.Shape(0);
    auto n_columns = t.Shape(1);
    common::ParallelFor1d<kBlockSize>(n_rows, n_threads, [&](auto&& block) {
      for (std::size_t i = block.begin(); i < block.end(); ++i) {
        for (std::size_t j = 0; j < n_columns; ++j) {
          fn(i, j);
        }
      }
    });
  } else {
    common::ParallelFor1d<kBlockSize>(t.Size(), n_threads, [&](auto&& block) {
      for (std::size_t i = block.begin(); i < block.end(); ++i) {
        std::apply(fn, linalg::UnravelIndex(i, t.Shape()));
      }
    });
  }
}
}  // namespace cpu_impl

template <typename T, std::int32_t D>
auto cbegin(TensorView<T, D> const& v) {  // NOLINT
  auto it = common::MakeIndexTransformIter([&](std::size_t i) -> std::remove_cv_t<T> const& {
    return std::apply(v, linalg::UnravelIndex(i, v.Shape()));
  });
  return it;
}

template <typename T, std::int32_t D>
auto cend(TensorView<T, D> const& v) {  // NOLINT
  return cbegin(v) + v.Size();
}

template <typename T, std::int32_t D>
auto begin(TensorView<T, D>& v) {  // NOLINT
  auto it = common::MakeIndexTransformIter(
      [&](std::size_t i) -> T& { return std::apply(v, linalg::UnravelIndex(i, v.Shape())); });
  return it;
}

template <typename T, std::int32_t D>
auto end(TensorView<T, D>& v) {  // NOLINT
  return begin(v) + v.Size();
}

/**
 * @brief Elementwise kernel without a return type.
 *
 * @tparam T  Element type of the input array.
 * @tparam D  Number of dimension of the input array.
 * @tparam Fn Transformation function.
 *
 * @param t  Input array.
 * @param fn Transformation function.
 */
#if defined(__CUDACC__)
template <typename T, std::int32_t D, typename Fn>
void ElementWiseKernel(Context const* ctx, TensorView<T, D> t, Fn&& fn) {
  ctx->DispatchDevice(
      [&] { cpu_impl::ElementWiseKernel(t, ctx->Threads(), std::forward<Fn>(fn)); },
      [&] { cuda_impl::ElementWiseKernel(t, std::forward<Fn>(fn), ctx->CUDACtx()->Stream()); });
}
#elif defined(XGBOOST_USE_SYCL)
template <typename T, std::int32_t D, typename Fn>
void ElementWiseKernel(Context const* ctx, TensorView<T, D> t, Fn&& fn) {
  ctx->DispatchDevice([&] { cpu_impl::ElementWiseKernel(t, ctx->Threads(), std::forward<Fn>(fn)); },
                      [&] { LOG(FATAL) << "Invalid TU"; },
                      [&] { ::xgboost::sycl::linalg::ElementWiseKernel(t, std::forward<Fn>(fn)); });
}
#else
template <typename T, std::int32_t D, typename Fn>
void ElementWiseKernel(Context const* ctx, TensorView<T, D> t, Fn&& fn) {
  ctx->DispatchDevice([&] { cpu_impl::ElementWiseKernel(t, ctx->Threads(), std::forward<Fn>(fn)); },
                      [&] { LOG(FATAL) << "Invalid TU"; });
}
#endif

/**
 * @brief Elementwise transform, with element index and the element itself as input.
 *
 * @tparam T  Element type of the input array.
 * @tparam D  Number of dimension of the input array.
 * @tparam Fn Transformation function, must return type T.
 *
 * @param t  Input array.
 * @param fn Transformation function, must return type T.
 */
#if defined(__CUDACC__)
template <typename T, std::int32_t D, typename Fn>
void TransformIdxKernel(Context const* ctx, TensorView<T, D> t, Fn&& fn) {
  ctx->DispatchDevice([&] { cpu_impl::TransformIdxKernel(t, ctx->Threads(), fn); },
                      [&] { cuda_impl::TransformIdxKernel(ctx, t, std::forward<Fn>(fn)); });
}
#elif defined(XGBOOST_USE_SYCL)
template <typename T, std::int32_t D, typename Fn>
void TransformIdxKernel(Context const* ctx, TensorView<T, D> t, Fn&& fn) {
  ctx->DispatchDevice([&] { cpu_impl::TransformIdxKernel(t, ctx->Threads(), fn); },
                      [&] { LOG(FATAL) << "Invalid TU."; },
                      [&] {
                        static_assert(D == 1, "Not implemented.");
                        sycl::linalg::ElementWiseKernel(
                            t, [=](std::size_t i) mutable { t(i) = fn(i, t(i)); });
                      });
}
#else
template <typename T, std::int32_t D, typename Fn>
void TransformIdxKernel(Context const* ctx, TensorView<T, D> t, Fn&& fn) {
  ctx->DispatchDevice([&] { cpu_impl::TransformIdxKernel(t, ctx->Threads(), fn); },
                      [&] { LOG(FATAL) << "Invalid TU."; });
}
#endif

/**
 * @brief Elementwise transform, with the element itself as input. Rest is the same as @ref
 * TransformIdxKernel
 */
#if defined(__CUDACC__)
template <typename T, std::int32_t D, typename Fn>
void TransformKernel(Context const* ctx, TensorView<T, D> t, Fn&& fn) {
  ctx->DispatchDevice([&] { cpu_impl::TransformKernel(t, ctx->Threads(), fn); },
                      [&] { cuda_impl::TransformKernel(ctx, t, std::forward<Fn>(fn)); });
}
#elif defined(XGBOOST_USE_SYCL)
template <typename T, std::int32_t D, typename Fn>
void TransformKernel(Context const* ctx, TensorView<T, D> t, Fn&& fn) {
  ctx->DispatchDevice([&] { cpu_impl::TransformKernel(t, ctx->Threads(), fn); },
                      [&] { LOG(FATAL) << "Invalid TU."; },
                      [&] {
                        static_assert(D == 1, "Not implemented.");
                        sycl::linalg::ElementWiseKernel(
                            t, [=](std::size_t i) mutable { t(i) = fn(t(i)); });
                      });
}
#else
template <typename T, std::int32_t D, typename Fn>
void TransformKernel(Context const* ctx, TensorView<T, D> t, Fn&& fn) {
  ctx->DispatchDevice([&] { cpu_impl::TransformKernel(t, ctx->Threads(), fn); },
                      [&] { LOG(FATAL) << "Invalid TU."; });
}
#endif

// vector-scalar multiplication
inline void VecScaMul(Context const* ctx, linalg::VectorView<float> x, double mul) {
  CHECK_EQ(x.Device().ordinal, ctx->Device().ordinal);
  TransformKernel(ctx, x, [=] XGBOOST_DEVICE(float v) { return v * mul; });
}

// vector-scalar division
inline void VecScaDiv(Context const* ctx, linalg::VectorView<float> x, double div) {
  return VecScaMul(ctx, x, 1.0 / div);
}

inline void LogE(Context const* ctx, linalg::VectorView<float> x) {
  CHECK_EQ(x.Device().ordinal, ctx->Device().ordinal);
  TransformKernel(ctx, x, [=] XGBOOST_DEVICE(float v) { return log(v); });
}

template <typename T, std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
void SaveVector(linalg::Vector<T> const& in, Json* p_out) {
  ::xgboost::SaveVector(in.Data()->HostVector(), p_out);
}

template <typename T, std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
void LoadVector(Json const& in, linalg::Vector<T>* out) {
  ::xgboost::LoadVector(in, &out->Data()->HostVector());
}

void SmallHistogram(Context const* ctx, linalg::MatrixView<float const> indices,
                    common::OptionalWeights const& weights, linalg::VectorView<float> bins);
}  // namespace xgboost::linalg
#endif  // XGBOOST_COMMON_LINALG_OP_H_
