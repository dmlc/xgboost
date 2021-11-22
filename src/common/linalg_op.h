/*!
 * Copyright 2021 by XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_LINALG_OP_H_
#define XGBOOST_COMMON_LINALG_OP_H_
#include "threading_utils.h"
#include "xgboost/linalg.h"

namespace xgboost {
namespace linalg {
template <typename T, int32_t D, typename Fn>
void ElementWiseKernelHost(linalg::TensorView<T, D> t, int32_t n_threads, Fn&& fn) {
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
}  // namespace linalg
}  // namespace xgboost
#endif  // XGBOOST_COMMON_LINALG_OP_H_
