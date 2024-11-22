/**
 * Copyright 2021-2023, XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_LINALG_OP_H_
#define XGBOOST_COMMON_LINALG_OP_H_
#include <cstdint>  // std::int32_t
#include <type_traits>

#include "common.h"
#include "threading_utils.h"
#include "transform_iterator.h"  // MakeIndexTransformIter
#include "xgboost/context.h"     // Context
#include "xgboost/linalg.h"

namespace xgboost {
namespace linalg {
template <typename T, int32_t D, typename Fn>
void ElementWiseTransformHost(linalg::TensorView<T, D> t, int32_t n_threads, Fn&& fn) {
  if (t.Contiguous()) {
    auto ptr = t.Values().data();
    common::ParallelFor(t.Size(), n_threads, [&](size_t i) { ptr[i] = fn(i, ptr[i]); });
  } else {
    common::ParallelFor(t.Size(), n_threads, [&](size_t i) {
      auto& v = detail::Apply(t, linalg::UnravelIndex(i, t.Shape()));
      v = fn(i, v);
    });
  }
}

template <typename T, std::int32_t D, typename Fn>
void ElementWiseKernelHost(linalg::TensorView<T, D> t, std::int32_t n_threads, Fn &&fn) {
  if constexpr (D == 1) {
    common::ParallelFor(t.Size(), n_threads, [&](std::size_t i) { fn(i); });
  } else if (D == 2 && t.CContiguous() && t.Shape(0) > t.Shape(1) * 64) {
    // Heuristic. Tall, c-contiguous matrix,
    auto n_rows = t.Shape(0);
    auto n_columns = t.Shape(1);
    common::ParallelFor(n_rows, n_threads, [&](std::size_t i) {
      for (std::size_t j = 0; j < n_columns; ++j) {
        fn(i, j);
      }
    });
  } else {
    common::ParallelFor(t.Size(), n_threads, [&](std::size_t i) {
      auto idx = linalg::UnravelIndex(i, t.Shape());
      std::apply(fn, idx);
    });
  }
}

#if !defined(XGBOOST_USE_CUDA) && !defined(XGBOOST_USE_SYCL)
template <typename T, int32_t D, typename Fn>
void ElementWiseKernelDevice(linalg::TensorView<T, D>, Fn&&, void* = nullptr) {
  common::AssertGPUSupport();
}

template <typename T, int32_t D, typename Fn>
void ElementWiseTransformDevice(linalg::TensorView<T, D>, Fn&&, void* = nullptr) {
  common::AssertGPUSupport();
}

template <typename T, int32_t D, typename Fn>
void ElementWiseKernel(Context const* ctx, linalg::TensorView<T, D> t, Fn&& fn) {
  if (ctx->IsCUDA()) {
    common::AssertGPUSupport();
  }
  ElementWiseKernelHost(t, ctx->Threads(), fn);
}
#endif  // !defined(XGBOOST_USE_CUDA) && !defined(XGBOOST_USE_SYCL)

template <typename T, std::int32_t kDim>
auto cbegin(TensorView<T, kDim> const& v) {  // NOLINT
  auto it = common::MakeIndexTransformIter([&](size_t i) -> std::remove_cv_t<T> const& {
    return linalg::detail::Apply(v, linalg::UnravelIndex(i, v.Shape()));
  });
  return it;
}

template <typename T, std::int32_t kDim>
auto cend(TensorView<T, kDim> const& v) {  // NOLINT
  return cbegin(v) + v.Size();
}

template <typename T, std::int32_t kDim>
auto begin(TensorView<T, kDim>& v) {  // NOLINT
  auto it = common::MakeIndexTransformIter(
      [&](size_t i) -> T& { return linalg::detail::Apply(v, linalg::UnravelIndex(i, v.Shape())); });
  return it;
}

template <typename T, std::int32_t kDim>
auto end(TensorView<T, kDim>& v) {  // NOLINT
  return begin(v) + v.Size();
}
}  // namespace linalg
}  // namespace xgboost
#endif  // XGBOOST_COMMON_LINALG_OP_H_
