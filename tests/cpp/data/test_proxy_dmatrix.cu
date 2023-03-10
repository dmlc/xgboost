/**
 * Copyright 2020-2023 XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/host_device_vector.h>

#include <any>  // for any_cast
#include <memory>

#include "../../../src/data/device_adapter.cuh"
#include "../../../src/data/proxy_dmatrix.h"
#include "../helpers.h"

namespace xgboost::data {
TEST(ProxyDMatrix, DeviceData) {
  constexpr size_t kRows{100}, kCols{100};
  HostDeviceVector<float> storage;
  auto data = RandomDataGenerator(kRows, kCols, 0.5).Device(0).GenerateArrayInterface(&storage);
  std::vector<HostDeviceVector<float>> label_storage(1);
  auto labels =
      RandomDataGenerator(kRows, 1, 0).Device(0).GenerateColumnarArrayInterface(&label_storage);

  DMatrixProxy proxy;
  proxy.SetCUDAArray(data.c_str());
  proxy.SetInfo("label", labels.c_str());

  ASSERT_EQ(proxy.Adapter().type(), typeid(std::shared_ptr<CupyAdapter>));
  ASSERT_EQ(proxy.Info().labels.Size(), kRows);
  ASSERT_EQ(std::any_cast<std::shared_ptr<CupyAdapter>>(proxy.Adapter())->NumRows(), kRows);
  ASSERT_EQ(std::any_cast<std::shared_ptr<CupyAdapter>>(proxy.Adapter())->NumColumns(), kCols);

  std::vector<HostDeviceVector<float>> columnar_storage(kCols);
  data = RandomDataGenerator(kRows, kCols, 0)
             .Device(0)
             .GenerateColumnarArrayInterface(&columnar_storage);
  proxy.SetCUDAArray(data.c_str());
  ASSERT_EQ(proxy.Adapter().type(), typeid(std::shared_ptr<CudfAdapter>));
  ASSERT_EQ(std::any_cast<std::shared_ptr<CudfAdapter>>(proxy.Adapter())->NumRows(), kRows);
  ASSERT_EQ(std::any_cast<std::shared_ptr<CudfAdapter>>(proxy.Adapter())->NumColumns(), kCols);
}
}  // namespace xgboost::data
