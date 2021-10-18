/*!
 * Copyright 2021-2022 by XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_LINALG_OP_H_
#define XGBOOST_COMMON_LINALG_OP_H_
#include <type_traits>

#include "threading_utils.h"
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

template <typename T, int32_t D, typename Fn>
void ElementWiseKernelHost(linalg::TensorView<T, D> t, int32_t n_threads, Fn&& fn) {
  static_assert(std::is_void<std::result_of_t<Fn(size_t, T&)>>::value,
                "For function with return, use transform instead.");
  if (t.Contiguous()) {
    auto ptr = t.Values().data();
    common::ParallelFor(t.Size(), n_threads, [&](size_t i) { fn(i, ptr[i]); });
  } else {
    common::ParallelFor(t.Size(), n_threads, [&](size_t i) {
      auto& v = detail::Apply(t, linalg::UnravelIndex(i, t.Shape()));
      fn(i, v);
    });
  }
}
}  // namespace linalg
}  // namespace xgboost
#endif  // XGBOOST_COMMON_LINALG_OP_H_
