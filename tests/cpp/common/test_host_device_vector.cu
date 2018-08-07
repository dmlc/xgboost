/*!
 * Copyright 2018 XGBoost contributors
 */

#include <gtest/gtest.h>
#include "../../../src/common/host_device_vector.h"
#include "../../../src/common/device_helpers.cuh"

namespace xgboost {
namespace common {

TEST(HostDeviceVector, Span) {
  HostDeviceVector<float> vec {1.0f, 2.0f, 3.0f, 4.0f};
  vec.Reshard(GPUSet{0, 1});
  auto span = vec.DeviceSpan(0);
  ASSERT_EQ(vec.Size(), span.size());
  ASSERT_EQ(vec.DevicePointer(0), span.data());
}

}  // namespace common
}  // namespace xgboost

