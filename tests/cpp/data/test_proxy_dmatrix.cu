/**
 * Copyright 2020-2025, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/host_device_vector.h>

#include <any>     // for any_cast
#include <memory>  // for shared_ptr
#include <vector>  // for vector

#include "../../../src/data/device_adapter.cuh"
#include "../../../src/data/proxy_dmatrix.h"
#include "../helpers.h"
#include "xgboost/host_device_vector.h"  // for HostDeviceVector

namespace xgboost::data {
TEST(ProxyDMatrix, DeviceData) {
  constexpr size_t kRows{100}, kCols{100};
  HostDeviceVector<float> storage;
  auto data =
      RandomDataGenerator(kRows, kCols, 0.5).Device(FstCU()).GenerateArrayInterface(&storage);
  std::vector<HostDeviceVector<float>> label_storage(1);
  auto labels = RandomDataGenerator(kRows, 1, 0)
                    .Device(FstCU())
                    .GenerateColumnarArrayInterface(&label_storage);

  DMatrixProxy proxy;
  proxy.SetCudaArray(data.c_str());
  proxy.SetInfo("label", labels.c_str());

  ASSERT_EQ(proxy.Adapter().type(), typeid(std::shared_ptr<CupyAdapter>));
  ASSERT_EQ(proxy.Info().labels.Size(), kRows);
  ASSERT_EQ(std::any_cast<std::shared_ptr<CupyAdapter>>(proxy.Adapter())->NumRows(), kRows);
  ASSERT_EQ(std::any_cast<std::shared_ptr<CupyAdapter>>(proxy.Adapter())->NumColumns(), kCols);

  std::vector<HostDeviceVector<float>> columnar_storage(kCols);
  data = RandomDataGenerator(kRows, kCols, 0)
             .Device(FstCU())
             .GenerateColumnarArrayInterface(&columnar_storage);
  proxy.SetCudaColumnar(data.c_str());
  ASSERT_EQ(proxy.Adapter().type(), typeid(std::shared_ptr<CudfAdapter>));
  ASSERT_EQ(std::any_cast<std::shared_ptr<CudfAdapter>>(proxy.Adapter())->NumRows(), kRows);
  ASSERT_EQ(std::any_cast<std::shared_ptr<CudfAdapter>>(proxy.Adapter())->NumColumns(), kCols);
}
}  // namespace xgboost::data
