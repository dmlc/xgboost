/**
 * Copyright 2021-2025, XGBoost Contributors
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

#if !defined(XGBOOST_USE_CUDA) && !defined(XGBOOST_USE_SYCL)

#include "common.h"           // for AssertGPUSupport
#include "xgboost/context.h"  // for Context

#endif  // !defined(XGBOOST_USE_CUDA) && !defined(XGBOOST_USE_SYCL)

namespace xgboost::common {
struct OptionalWeights;
}

namespace xgboost::linalg {
template <typename T, int32_t D, typename Fn>
void ElementWiseTransformHost(linalg::TensorView<T, D> t, int32_t n_threads, Fn&& fn) {
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
void ElementWiseKernelHost(linalg::TensorView<T, D> t, std::int32_t n_threads, Fn&& fn) {
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
  auto it = common::MakeIndexTransformIter([&](std::size_t i) -> std::remove_cv_t<T> const& {
    return std::apply(v, linalg::UnravelIndex(i, v.Shape()));
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
      [&](std::size_t i) -> T& { return std::apply(v, linalg::UnravelIndex(i, v.Shape())); });
  return it;
}

template <typename T, std::int32_t kDim>
auto end(TensorView<T, kDim>& v) {  // NOLINT
  return begin(v) + v.Size();
}

namespace cuda_impl {
void VecScaMul(Context const* ctx, linalg::VectorView<float> x, double mul);
}  // namespace cuda_impl

namespace sycl_impl {
void VecScaMul(Context const* ctx, linalg::VectorView<float> x, double mul);
}  // namespace sycl_impl

// vector-scalar multiplication
inline void VecScaMul(Context const* ctx, linalg::VectorView<float> x, double mul) {
  CHECK_EQ(x.Device().ordinal, ctx->Device().ordinal);
  if (x.Device().IsCUDA()) {
#if defined(XGBOOST_USE_CUDA)
    cuda_impl::VecScaMul(ctx, x, mul);
#else
    common::AssertGPUSupport();
#endif
  } else if (x.Device().IsSycl()) {
#if defined(XGBOOST_USE_SYCL)
    sycl_impl::VecScaMul(ctx, x, mul);
#else
    common::AssertSYCLSupport();
#endif
  } else {
    constexpr std::size_t kBlockSize = 2048;
    common::ParallelFor1d<kBlockSize>(x.Size(), ctx->Threads(), [&](auto&& block) {
      for (auto i = block.begin(); i < block.end(); ++i) {
        x(i) *= mul;
      }
    });
  }
}

// vector-scalar division
inline void VecScaDiv(Context const* ctx, linalg::VectorView<float> x, double div) {
  return VecScaMul(ctx, x, 1.0 / div);
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
