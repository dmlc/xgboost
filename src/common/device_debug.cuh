/**
 * Copyright 2025, XGBoost contributors
 */
#include <cstddef>   // for size_t
#include <iostream>  // for cout
#include <vector>    // for vector

#include "common.h"
#include "device_helpers.cuh"     // for CopyDeviceSpanToVector
#include "xgboost/span.h"         // for Span
#include "xgboost/string_view.h"  // for StringView

namespace xgboost::debug {
// debug::SyncDevice(__FILE__, __LINE__);
inline void SyncDevice(char const *file = __builtin_FILE(), int32_t line = __builtin_LINE()) {
  {
    auto err = cudaDeviceSynchronize();
    dh::ThrowOnCudaError(err, file, line);
  }
  {
    auto err = cudaGetLastError();
    dh::ThrowOnCudaError(err, file, line);
  }
}

template <typename T>
void PrintDeviceSpan(common::Span<T> values, StringView name) {
  std::cout << name << std::endl;
  std::vector<std::remove_cv_t<T>> h_values(values.size());
  dh::CopyDeviceSpanToVector(&h_values, values);
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0 && i % 16 == 0) {
      std::cout << std::endl;
    }
    std::cout << h_values[i] << ", ";
  }
  std::cout << std::endl;
}
}  // namespace xgboost::debug
