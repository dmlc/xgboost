/**
 * Copyright 2024, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include "../../../src/common/device_vector.cuh"
#include "xgboost/global_config.h"  // for GlobalConfigThreadLocalStore

namespace dh {
TEST(DeviceUVector, Basic) {
  GlobalMemoryLogger().Clear();
  std::int32_t verbosity{3};
  std::swap(verbosity, xgboost::GlobalConfigThreadLocalStore::Get()->verbosity);
  DeviceUVector<float> uvec;
  uvec.Resize(12);
  auto peak = GlobalMemoryLogger().PeakMemory();
  auto n_bytes = sizeof(decltype(uvec)::value_type) * uvec.size();
  ASSERT_EQ(peak, n_bytes);
  std::swap(verbosity, xgboost::GlobalConfigThreadLocalStore::Get()->verbosity);
}
}  // namespace dh
