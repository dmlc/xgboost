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

void InitHostDeviceVector(size_t n, const GPUDistribution& distribution,
                     HostDeviceVector<unsigned int> *v) {
  // create the vector
  GPUSet devices = distribution.Devices();
  v->Reshard(distribution);
  v->Resize(n);

  ASSERT_EQ(v->Size(), n);
  ASSERT_TRUE(v->Distribution() == distribution);
  ASSERT_TRUE(v->Devices() == devices);
  // ensure that the devices have read-write access
  for (int i = 0; i < devices.Size(); ++i) {
    ASSERT_TRUE(v->DeviceCanAccess(i, GPUAccess::kRead));
    ASSERT_TRUE(v->DeviceCanAccess(i, GPUAccess::kWrite));
  }
  // ensure that the host has no access
  ASSERT_FALSE(v->HostCanAccess(GPUAccess::kWrite));
  ASSERT_FALSE(v->HostCanAccess(GPUAccess::kRead));

  // fill in the data on the host
  std::vector<unsigned int>& data_h = v->HostVector();
  // ensure that the host has full access, while the devices have none
  ASSERT_TRUE(v->HostCanAccess(GPUAccess::kRead));
  ASSERT_TRUE(v->HostCanAccess(GPUAccess::kWrite));
  for (int i = 0; i < devices.Size(); ++i) {
    ASSERT_FALSE(v->DeviceCanAccess(i, GPUAccess::kRead));
    ASSERT_FALSE(v->DeviceCanAccess(i, GPUAccess::kWrite));
  }
  ASSERT_EQ(data_h.size(), n);
  std::copy_n(thrust::make_counting_iterator(0), n, data_h.begin());
}

void PlusOne(HostDeviceVector<unsigned int> *v) {
  int n_devices = v->Devices().Size();
  for (int i = 0; i < n_devices; ++i) {
    dh::safe_cuda(cudaSetDevice(i));
    thrust::transform(v->tbegin(i), v->tend(i), v->tbegin(i),
                      [=]__device__(unsigned int a){ return a + 1; });
  }
}

void CheckDevice(HostDeviceVector<unsigned int> *v,
                 const std::vector<size_t>& starts,
                 const std::vector<size_t>& sizes,
                 unsigned int first, GPUAccess access) {
  int n_devices = sizes.size();
  ASSERT_EQ(v->Devices().Size(), n_devices);
  for (int i = 0; i < n_devices; ++i) {
    ASSERT_EQ(v->DeviceSize(i), sizes.at(i));
    dh::safe_cuda(cudaSetDevice(i));
    ASSERT_TRUE(thrust::equal(v->tcbegin(i), v->tcend(i),
                              thrust::make_counting_iterator(first + starts[i])));
    ASSERT_TRUE(v->DeviceCanAccess(i, GPUAccess::kRead));
    // ensure that the device has at most the access specified by access
    ASSERT_EQ(v->DeviceCanAccess(i, GPUAccess::kWrite), access == GPUAccess::kWrite);
  }
  ASSERT_EQ(v->HostCanAccess(GPUAccess::kRead), access == GPUAccess::kRead);
  ASSERT_FALSE(v->HostCanAccess(GPUAccess::kWrite));
  for (int i = 0; i < n_devices; ++i) {
    dh::safe_cuda(cudaSetDevice(i));
    ASSERT_TRUE(thrust::equal(v->tbegin(i), v->tend(i),
                              thrust::make_counting_iterator(first + starts[i])));
    ASSERT_TRUE(v->DeviceCanAccess(i, GPUAccess::kRead));
    ASSERT_TRUE(v->DeviceCanAccess(i, GPUAccess::kWrite));
  }
  ASSERT_FALSE(v->HostCanAccess(GPUAccess::kRead));
  ASSERT_FALSE(v->HostCanAccess(GPUAccess::kWrite));
}

void CheckHost(HostDeviceVector<unsigned int> *v, GPUAccess access) {
  const std::vector<unsigned int>& data_h = access == GPUAccess::kWrite ?
    v->HostVector() : v->ConstHostVector();
  for (size_t i = 0; i < v->Size(); ++i) {
    ASSERT_EQ(data_h.at(i), i + 1);
  }
  ASSERT_TRUE(v->HostCanAccess(GPUAccess::kRead));
  ASSERT_EQ(v->HostCanAccess(GPUAccess::kWrite), access == GPUAccess::kWrite);
  size_t n_devices = v->Devices().Size();
  for (int i = 0; i < n_devices; ++i) {
    ASSERT_EQ(v->DeviceCanAccess(i, GPUAccess::kRead), access == GPUAccess::kRead);
    // the devices should have no write access
    ASSERT_FALSE(v->DeviceCanAccess(i, GPUAccess::kWrite));
  }
}

void TestHostDeviceVector
(size_t n, const GPUDistribution& distribution,
 const std::vector<size_t>& starts, const std::vector<size_t>& sizes) {
  HostDeviceVector<unsigned int> v;
  InitHostDeviceVector(n, distribution, &v);
  CheckDevice(&v, starts, sizes, 0, GPUAccess::kRead);
  PlusOne(&v);
  CheckDevice(&v, starts, sizes, 1, GPUAccess::kWrite);
  CheckHost(&v, GPUAccess::kRead);
  CheckHost(&v, GPUAccess::kWrite);
}

TEST(HostDeviceVector, TestBlock) {
  size_t n = 1001;
  int n_devices = 2;
  auto distribution = GPUDistribution::Block(GPUSet::Range(0, n_devices));
  std::vector<size_t> starts{0, 501};
  std::vector<size_t> sizes{501, 500};
  TestHostDeviceVector(n, distribution, starts, sizes);
}

TEST(HostDeviceVector, TestGranular) {
  size_t n = 3003;
  int n_devices = 2;
  auto distribution = GPUDistribution::Granular(GPUSet::Range(0, n_devices), 3);
  std::vector<size_t> starts{0, 1503};
  std::vector<size_t> sizes{1503, 1500};
  TestHostDeviceVector(n, distribution, starts, sizes);
}

TEST(HostDeviceVector, TestOverlap) {
  size_t n = 1001;
  int n_devices = 2;
  auto distribution = GPUDistribution::Overlap(GPUSet::Range(0, n_devices), 1);
  std::vector<size_t> starts{0, 500};
  std::vector<size_t> sizes{501, 501};
  TestHostDeviceVector(n, distribution, starts, sizes);
}

TEST(HostDeviceVector, TestExplicit) {
  size_t n = 1001;
  int n_devices = 2;
  std::vector<size_t> offsets{0, 550, 1001};
  auto distribution = GPUDistribution::Explicit(GPUSet::Range(0, n_devices), offsets);
  std::vector<size_t> starts{0, 550};
  std::vector<size_t> sizes{550, 451};
  TestHostDeviceVector(n, distribution, starts, sizes);
}

}  // namespace common
}  // namespace xgboost
