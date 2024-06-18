/**
 * Copyright 2021-2023, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/host_device_vector.h>
#include "../helpers.h"
#include "../../../src/data/array_interface.h"

namespace xgboost {

__global__ void SleepForTest(uint64_t *out, uint64_t duration) {
  auto start = clock64();
  auto t = 0;
  while (t < duration) {
    t = clock64() - start;
  }
  out[0] = t;
}

TEST(ArrayInterface, Stream) {
  size_t constexpr kRows = 10, kCols = 10;
  HostDeviceVector<float> storage;
  auto arr_str = RandomDataGenerator{kRows, kCols, 0}.GenerateArrayInterface(&storage);

  dh::CUDAStream stream;

  auto j_arr = Json::Load(StringView{arr_str});
  j_arr["stream"] = Integer(reinterpret_cast<int64_t>(stream.Handle()));
  Json::Dump(j_arr, &arr_str);

  dh::caching_device_vector<uint64_t> out(1, 0);
  std::uint64_t dur = 1e9;
  dh::LaunchKernel{1, 1, 0, stream.View()}(SleepForTest, out.data().get(), dur);
  ArrayInterface<2> arr(arr_str);

  auto t = out[0];
  CHECK_GE(t, dur);
}

TEST(ArrayInterface, Ptr) {
  std::vector<float> h_data(10);
  ASSERT_FALSE(ArrayInterfaceHandler::IsCudaPtr(h_data.data()));
  dh::safe_cuda(cudaGetLastError());

  dh::device_vector<float> d_data(10);
  ASSERT_TRUE(ArrayInterfaceHandler::IsCudaPtr(d_data.data().get()));
  dh::safe_cuda(cudaGetLastError());

  ASSERT_FALSE(ArrayInterfaceHandler::IsCudaPtr(nullptr));
  dh::safe_cuda(cudaGetLastError());
}
}  // namespace xgboost
