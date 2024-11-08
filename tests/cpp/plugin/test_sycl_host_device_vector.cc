/**
 * Copyright 2018-2024, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <numeric>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-W#pragma-messages"
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#include <xgboost/host_device_vector.h>
#pragma GCC diagnostic pop

#include "sycl_helpers.h"

namespace xgboost::common {
namespace {

void InitHostDeviceVector(size_t n, DeviceOrd device, HostDeviceVector<int> *v) {
  // create the vector
  v->SetDevice(device);
  v->Resize(n);

  ASSERT_EQ(v->Size(), n);
  ASSERT_EQ(v->Device(), device);
  // ensure that the device have read-write access
  ASSERT_TRUE(v->DeviceCanRead());
  ASSERT_TRUE(v->DeviceCanWrite());
  // ensure that the host has no access
  ASSERT_FALSE(v->HostCanRead());
  ASSERT_FALSE(v->HostCanWrite());

  // fill in the data on the host
  std::vector<int>& data_h = v->HostVector();
  // ensure that the host has full access, while the device have none
  ASSERT_TRUE(v->HostCanRead());
  ASSERT_TRUE(v->HostCanWrite());
  ASSERT_FALSE(v->DeviceCanRead());
  ASSERT_FALSE(v->DeviceCanWrite());
  ASSERT_EQ(data_h.size(), n);
  std::iota(data_h.begin(), data_h.end(), 0);
}

void PlusOne(HostDeviceVector<int> *v) {
  auto device = v->Device();
  sycl::TransformOnDeviceData(v->Device(), v->DevicePointer(), v->Size(), [=](size_t a){ return a + 1; });
  ASSERT_TRUE(v->DeviceCanWrite());
}

void CheckDevice(HostDeviceVector<int>* v,
                 size_t size,
                 unsigned int first,
                 GPUAccess access) {
  ASSERT_EQ(v->Size(), size);

  std::vector<int> desired_data(size);
  std::iota(desired_data.begin(), desired_data.end(), first);
  sycl::VerifyOnDeviceData(v->Device(), v->ConstDevicePointer(), desired_data.data(), size);
  ASSERT_TRUE(v->DeviceCanRead());
  // ensure that the device has at most the access specified by access
  ASSERT_EQ(v->DeviceCanWrite(), access == GPUAccess::kWrite);
  ASSERT_EQ(v->HostCanRead(), access == GPUAccess::kRead);
  ASSERT_FALSE(v->HostCanWrite());

  sycl::VerifyOnDeviceData(v->Device(), v->DevicePointer(), desired_data.data(), size);
  ASSERT_TRUE(v->DeviceCanRead());
  ASSERT_TRUE(v->DeviceCanWrite());
  ASSERT_FALSE(v->HostCanRead());
  ASSERT_FALSE(v->HostCanWrite());
}

void CheckHost(HostDeviceVector<int> *v, GPUAccess access) {
  const std::vector<int>& data_h = access == GPUAccess::kNone ?
    v->HostVector() : v->ConstHostVector();
  for (size_t i = 0; i < v->Size(); ++i) {
    ASSERT_EQ(data_h.at(i), i + 1);
  }
  ASSERT_TRUE(v->HostCanRead());
  ASSERT_EQ(v->HostCanWrite(), access == GPUAccess::kNone);
  ASSERT_EQ(v->DeviceCanRead(), access == GPUAccess::kRead);
  // the devices should have no write access
  ASSERT_FALSE(v->DeviceCanWrite());
}

void TestHostDeviceVector(size_t n, DeviceOrd device) {
  HostDeviceVector<int> v;
  InitHostDeviceVector(n, device, &v);
  CheckDevice(&v, n, 0, GPUAccess::kRead);
  PlusOne(&v);
  CheckDevice(&v, n, 1, GPUAccess::kWrite);
  CheckHost(&v, GPUAccess::kRead);
  CheckHost(&v, GPUAccess::kNone);
}

TEST(SyclHostDeviceVector, Basic) {
  size_t n = 1001;
  DeviceOrd device = DeviceOrd::SyclDefault();
  TestHostDeviceVector(n, device);
}

TEST(SyclHostDeviceVector, Copy) {
  size_t n = 1001;
  auto device = DeviceOrd::SyclDefault();

  HostDeviceVector<int> v;
  {
    // a separate scope to ensure that v1 is gone before further checks
    HostDeviceVector<int> v1;
    InitHostDeviceVector(n, device, &v1);
    v.Resize(v1.Size());
    v.Copy(v1);
  }
  CheckDevice(&v, n, 0, GPUAccess::kRead);
  PlusOne(&v);
  CheckDevice(&v, n, 1, GPUAccess::kWrite);
  CheckHost(&v, GPUAccess::kRead);
  CheckHost(&v, GPUAccess::kNone);
}

TEST(SyclHostDeviceVector, Fill) {
  size_t n = 1001;
  auto device = DeviceOrd::SyclDefault();

  int val = 42;
  HostDeviceVector<int> v;
  v.SetDevice(device);
  v.Resize(n);

  ASSERT_TRUE(v.DeviceCanWrite());
  v.Fill(val);

  ASSERT_FALSE(v.HostCanRead());
  ASSERT_FALSE(v.HostCanWrite());
  ASSERT_TRUE(v.DeviceCanRead());
  ASSERT_TRUE(v.DeviceCanWrite());

  std::vector<int> desired_data(n, val);
  sycl::VerifyOnDeviceData(v.Device(), v.ConstDevicePointer(), desired_data.data(), n);
}

TEST(SyclHostDeviceVector, Extend) {
  size_t n0 = 1001;
  size_t n1 = 17;
  auto device = DeviceOrd::SyclDefault();

  int val = 42;
  HostDeviceVector<int> v0;
  v0.SetDevice(device);
  v0.Resize(n0);
  v0.Fill(val);

  HostDeviceVector<int> v1;
  v1.SetDevice(device);
  v1.Resize(n1);
  v1.Fill(val);

  v0.Extend(v1);
  {
    std::vector<int> desired_data(n0+n1, val);
    sycl::VerifyOnDeviceData(v0.Device(), v0.ConstDevicePointer(), desired_data.data(), n0+n1);
  }
  v1.Extend(v0);
  {
    std::vector<int> desired_data(n0+2*n1, val);
    sycl::VerifyOnDeviceData(v1.Device(), v1.ConstDevicePointer(), desired_data.data(), n0+2*n1);
  }
}

TEST(SyclHostDeviceVector, SetDevice) {
  std::vector<int> h_vec (2345);
  for (size_t i = 0; i < h_vec.size(); ++i) {
    h_vec[i] = i;
  }
  HostDeviceVector<int> vec (h_vec);
  auto device = DeviceOrd::SyclDefault();

  vec.SetDevice(device);
  ASSERT_EQ(vec.Size(), h_vec.size());
  auto span = vec.DeviceSpan();  // sync to device

  vec.SetDevice(DeviceOrd::CPU());  // pull back to cpu.
  ASSERT_EQ(vec.Size(), h_vec.size());
  ASSERT_EQ(vec.Device(), DeviceOrd::CPU());

  auto h_vec_1 = vec.HostVector();
  ASSERT_TRUE(std::equal(h_vec_1.cbegin(), h_vec_1.cend(), h_vec.cbegin()));
}

TEST(SyclHostDeviceVector, Span) {
  HostDeviceVector<float> vec {1.0f, 2.0f, 3.0f, 4.0f};
  vec.SetDevice(DeviceOrd::SyclDefault());
  auto span = vec.DeviceSpan();
  ASSERT_EQ(vec.Size(), span.size());
  ASSERT_EQ(vec.DevicePointer(), span.data());
  auto const_span = vec.ConstDeviceSpan();
  ASSERT_EQ(vec.Size(), const_span.size());
  ASSERT_EQ(vec.ConstDevicePointer(), const_span.data());

  auto h_span = vec.ConstHostSpan();
  ASSERT_TRUE(vec.HostCanRead());
  ASSERT_FALSE(vec.HostCanWrite());
  ASSERT_EQ(h_span.size(), vec.Size());
  ASSERT_EQ(h_span.data(), vec.ConstHostPointer());

  h_span = vec.HostSpan();
  ASSERT_TRUE(vec.HostCanWrite());
}

TEST(SyclHostDeviceVector, Empty) {
  HostDeviceVector<float> vec {1.0f, 2.0f, 3.0f, 4.0f};
  HostDeviceVector<float> another { std::move(vec) };
  ASSERT_FALSE(another.Empty());
  ASSERT_TRUE(vec.Empty());
}

TEST(SyclHostDeviceVector, Resize) {
  auto check = [&](HostDeviceVector<float> const& vec) {
    auto const& h_vec = vec.ConstHostSpan();
    for (std::size_t i = 0; i < 4; ++i) {
      ASSERT_EQ(h_vec[i], i + 1);
    }
    for (std::size_t i = 4; i < vec.Size(); ++i) {
      ASSERT_EQ(h_vec[i], 3.0);
    }
  };
  {
    HostDeviceVector<float> vec{1.0f, 2.0f, 3.0f, 4.0f};
    vec.SetDevice(DeviceOrd::SyclDefault());
    vec.ConstDeviceSpan();
    ASSERT_TRUE(vec.DeviceCanRead());
    ASSERT_FALSE(vec.DeviceCanWrite());
    vec.DeviceSpan();
    vec.Resize(7, 3.0f);
    ASSERT_TRUE(vec.DeviceCanWrite());
    check(vec);
  }
  {
    HostDeviceVector<float> vec{{1.0f, 2.0f, 3.0f, 4.0f}, DeviceOrd::SyclDefault()};
    ASSERT_TRUE(vec.DeviceCanWrite());
    vec.Resize(7, 3.0f);
    ASSERT_TRUE(vec.DeviceCanWrite());
    check(vec);
  }
  {
    HostDeviceVector<float> vec{1.0f, 2.0f, 3.0f, 4.0f};
    ASSERT_TRUE(vec.HostCanWrite());
    vec.Resize(7, 3.0f);
    ASSERT_TRUE(vec.HostCanWrite());
    check(vec);
  }
}
}
}  // namespace xgboost::common
