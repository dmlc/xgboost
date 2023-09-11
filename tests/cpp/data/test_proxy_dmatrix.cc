/**
 * Copyright 2021-2023, XGBoost contributors
 */
#include <gtest/gtest.h>

#include "../../../src/data/adapter.h"
#include "../../../src/data/proxy_dmatrix.h"
#include "../helpers.h"

namespace xgboost::data {
TEST(ProxyDMatrix, HostData) {
  DMatrixProxy proxy;
  size_t constexpr kRows = 100, kCols = 10;
  std::vector<HostDeviceVector<float>> label_storage(1);

  HostDeviceVector<float> storage;
  auto data =
      RandomDataGenerator(kRows, kCols, 0.5).Device(FstCU()).GenerateArrayInterface(&storage);

  proxy.SetArrayData(data.c_str());

  auto n_samples = HostAdapterDispatch(&proxy, [](auto const &value) { return value.Size(); });
  ASSERT_EQ(n_samples, kRows);
  auto n_features = HostAdapterDispatch(&proxy, [](auto const &value) { return value.NumCols(); });
  ASSERT_EQ(n_features, kCols);
}
}  // namespace xgboost::data
