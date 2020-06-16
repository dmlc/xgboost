#include <gtest/gtest.h>
#include <xgboost/host_device_vector.h>
#include <memory>
#include "../helpers.h"
#include "../../../src/data/device_adapter.cuh"
#include "../../../src/data/proxy_dmatrix.h"

namespace xgboost {
namespace data {
TEST(ProxyDMatrix, Basic) {
  constexpr size_t kRows{100}, kCols{100};
  HostDeviceVector<float> storage;
  auto data = RandomDataGenerator(kRows, kCols, 0.5)
                  .Device(0)
                  .GenerateArrayInterface(&storage);
  std::vector<HostDeviceVector<float>> label_storage(1);
  auto labels = RandomDataGenerator(kRows, 1, 0)
                    .Device(0)
                    .GenerateColumnarArrayInterface(&label_storage);

  DMatrixProxy proxy;
  proxy.FromCudaArray(data);
  proxy.SetInfo("label", labels.c_str());

  ASSERT_EQ(proxy.Adapter().type(), typeid(std::shared_ptr<CupyAdapter>));
  ASSERT_EQ(proxy.Info().labels_.Size(), kRows);
  ASSERT_EQ(dmlc::get<std::shared_ptr<CupyAdapter>>(proxy.Adapter())->NumRows(),
            kRows);
  ASSERT_EQ(
      dmlc::get<std::shared_ptr<CupyAdapter>>(proxy.Adapter())->NumColumns(),
      kCols);

  std::vector<HostDeviceVector<float>> columnar_storage(kCols);
  data = RandomDataGenerator(kRows, kCols, 0)
                    .Device(0)
                    .GenerateColumnarArrayInterface(&columnar_storage);
  proxy.FromCudaColumnar(data);
  ASSERT_EQ(proxy.Adapter().type(), typeid(std::shared_ptr<CudfAdapter>));
  ASSERT_EQ(dmlc::get<std::shared_ptr<CudfAdapter>>(proxy.Adapter())->NumRows(),
            kRows);
  ASSERT_EQ(
      dmlc::get<std::shared_ptr<CudfAdapter>>(proxy.Adapter())->NumColumns(),
      kCols);
}
}  // namespace data
}  // namespace xgboost
