/**
 * Copyright 2021-2025, XGBoost contributors
 */
#include <gtest/gtest.h>

#include <cstddef>  // for size_t
#include <vector>   // for vector

#include "../../../src/data/proxy_dmatrix.h"
#include "../helpers.h"
#include "xgboost/host_device_vector.h"  // for HostDeviceVector

namespace xgboost::data {
TEST(ProxyDMatrix, HostData) {
  DMatrixProxy proxy;
  std::size_t constexpr kRows = 100, kCols = 10;
  std::vector<HostDeviceVector<float>> label_storage(1);

  HostDeviceVector<float> storage;
  auto data =
      RandomDataGenerator(kRows, kCols, 0.5).Device(FstCU()).GenerateArrayInterface(&storage);

  proxy.SetArray(data.c_str());
  using cpu_impl::DispatchAny;

  auto n_samples = DispatchAny(&proxy, [](auto const &value) { return value.Size(); });
  ASSERT_EQ(n_samples, kRows);
  auto n_features = DispatchAny(&proxy, [](auto const &value) { return value.NumCols(); });
  ASSERT_EQ(n_features, kCols);
}
}  // namespace xgboost::data
