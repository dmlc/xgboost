/*!
 * Copyright 2021 by XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_LINALG_OP_CUH_
#define XGBOOST_COMMON_LINALG_OP_CUH_
#include "device_helpers.cuh"
#include "xgboost/linalg.h"

namespace xgboost {
namespace linalg {
template <typename T, int32_t D, typename Fn>
void ElementWiseKernelDevice(linalg::TensorView<T, D> t, Fn&& fn, cudaStream_t s = nullptr) {
  if (t.Contiguous()) {
    auto ptr = t.Values().data();
    dh::LaunchN(t.Size(), s, [=] __device__(size_t i) { ptr[i] = fn(i, ptr[i]); });
  } else {
    dh::LaunchN(t.Size(), s, [=] __device__(size_t i) mutable {
      T& v = detail::Apply(t, linalg::UnravelIndex(i, t.Shape()));
      v = fn(i, v);
    });
  }
}
}  // namespace linalg
}  // namespace xgboost
#endif  // XGBOOST_COMMON_LINALG_OP_CUH_
