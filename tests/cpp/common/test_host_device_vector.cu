/*!
 * Copyright 2018 XGBoost contributors
 */

#include <gtest/gtest.h>
#include <thrust/equal.h>
#include <thrust/iterator/counting_iterator.h>

#include "../../../src/common/device_helpers.cuh"
#include "../../../src/common/host_device_vector.h"

namespace xgboost {
namespace common {

void SetDevice(int device) {
  int n_devices;
  dh::safe_cuda(cudaGetDeviceCount(&n_devices));
  device %= n_devices;
  dh::safe_cuda(cudaSetDevice(device));
}

struct HostDeviceVectorSetDeviceHandler {
  template <typename Functor>
  explicit HostDeviceVectorSetDeviceHandler(Functor f) {
    SetCudaSetDeviceHandler(f);
  }

  ~HostDeviceVectorSetDeviceHandler() {
    SetCudaSetDeviceHandler(nullptr);
  }
};

void InitHostDeviceVector(size_t n, int device, HostDeviceVector<int> *v) {
  // create the vector
  v->Shard(device);
  v->Resize(n);

  ASSERT_EQ(v->Size(), n);
  ASSERT_EQ(v->DeviceIdx(), device);
  // ensure that the device have read-write access
  ASSERT_TRUE(v->DeviceCanAccess(device, GPUAccess::kRead));
  ASSERT_TRUE(v->DeviceCanAccess(device, GPUAccess::kWrite));
  // ensure that the host has no access
  ASSERT_FALSE(v->HostCanAccess(GPUAccess::kWrite));
  ASSERT_FALSE(v->HostCanAccess(GPUAccess::kRead));

  // fill in the data on the host
  std::vector<int>& data_h = v->HostVector();
  // ensure that the host has full access, while the device have none
  ASSERT_TRUE(v->HostCanAccess(GPUAccess::kRead));
  ASSERT_TRUE(v->HostCanAccess(GPUAccess::kWrite));
  ASSERT_FALSE(v->DeviceCanAccess(device, GPUAccess::kRead));
  ASSERT_FALSE(v->DeviceCanAccess(device, GPUAccess::kWrite));
  ASSERT_EQ(data_h.size(), n);
  std::copy_n(thrust::make_counting_iterator(0), n, data_h.begin());
}

void PlusOne(HostDeviceVector<int> *v) {
  int device = v->DeviceIdx();
  SetDevice(device);
  thrust::transform(v->tbegin(0), v->tend(0), v->tbegin(0),
                    [=]__device__(unsigned int a){ return a + 1; });
}

void CheckDevice(HostDeviceVector<int> *v,
                 const std::vector<size_t>& starts,
                 const std::vector<size_t>& sizes,
                 unsigned int first, GPUAccess access) {
  int n_devices = sizes.size();
  ASSERT_EQ(n_devices, 1);
  for (int i = 0; i < n_devices; ++i) {
    ASSERT_EQ(v->DeviceSize(i), sizes.at(i));
    SetDevice(i);
    ASSERT_TRUE(thrust::equal(v->tcbegin(i), v->tcend(i),
                              thrust::make_counting_iterator(first + starts[i])));
    ASSERT_TRUE(v->DeviceCanAccess(i, GPUAccess::kRead));
    // ensure that the device has at most the access specified by access
    ASSERT_EQ(v->DeviceCanAccess(i, GPUAccess::kWrite), access == GPUAccess::kWrite);
  }
  ASSERT_EQ(v->HostCanAccess(GPUAccess::kRead), access == GPUAccess::kRead);
  ASSERT_FALSE(v->HostCanAccess(GPUAccess::kWrite));
  for (int i = 0; i < n_devices; ++i) {
    SetDevice(i);
    ASSERT_TRUE(thrust::equal(v->tbegin(i), v->tend(i),
                              thrust::make_counting_iterator(first + starts[i])));
    ASSERT_TRUE(v->DeviceCanAccess(i, GPUAccess::kRead));
    ASSERT_TRUE(v->DeviceCanAccess(i, GPUAccess::kWrite));
  }
  ASSERT_FALSE(v->HostCanAccess(GPUAccess::kRead));
  ASSERT_FALSE(v->HostCanAccess(GPUAccess::kWrite));
}

void CheckHost(HostDeviceVector<int> *v, GPUAccess access) {
  const std::vector<int>& data_h = access == GPUAccess::kWrite ?
    v->HostVector() : v->ConstHostVector();
  for (size_t i = 0; i < v->Size(); ++i) {
    ASSERT_EQ(data_h.at(i), i + 1);
  }
  ASSERT_TRUE(v->HostCanAccess(GPUAccess::kRead));
  ASSERT_EQ(v->HostCanAccess(GPUAccess::kWrite), access == GPUAccess::kWrite);
  size_t n_devices = 1;
  for (int i = 0; i < n_devices; ++i) {
    ASSERT_EQ(v->DeviceCanAccess(i, GPUAccess::kRead), access == GPUAccess::kRead);
    // the devices should have no write access
    ASSERT_FALSE(v->DeviceCanAccess(i, GPUAccess::kWrite));
  }
}

void TestHostDeviceVector
(size_t n, int device,
 const std::vector<size_t>& starts, const std::vector<size_t>& sizes) {
  HostDeviceVectorSetDeviceHandler hdvec_dev_hndlr(SetDevice);
  HostDeviceVector<int> v;
  InitHostDeviceVector(n, device, &v);
  CheckDevice(&v, starts, sizes, 0, GPUAccess::kRead);
  PlusOne(&v);
  CheckDevice(&v, starts, sizes, 1, GPUAccess::kWrite);
  CheckHost(&v, GPUAccess::kRead);
  CheckHost(&v, GPUAccess::kWrite);
}

TEST(HostDeviceVector, TestBlock) {
  size_t n = 1001;
  int device = 0;
  std::vector<size_t> starts{0};
  std::vector<size_t> sizes{1001};
  TestHostDeviceVector(n, device, starts, sizes);
}

TEST(HostDeviceVector, TestCopy) {
  size_t n = 1001;
  int device = 0;
  std::vector<size_t> starts{0};
  std::vector<size_t> sizes{1001};
  HostDeviceVectorSetDeviceHandler hdvec_dev_hndlr(SetDevice);

  HostDeviceVector<int> v;
  {
    // a separate scope to ensure that v1 is gone before further checks
    HostDeviceVector<int> v1;
    InitHostDeviceVector(n, device, &v1);
    v = v1;
  }
  CheckDevice(&v, starts, sizes, 0, GPUAccess::kRead);
  PlusOne(&v);
  CheckDevice(&v, starts, sizes, 1, GPUAccess::kWrite);
  CheckHost(&v, GPUAccess::kRead);
  CheckHost(&v, GPUAccess::kWrite);
}

TEST(HostDeviceVector, Shard) {
  std::vector<int> h_vec (2345);
  for (size_t i = 0; i < h_vec.size(); ++i) {
    h_vec[i] = i;
  }
  HostDeviceVector<int> vec (h_vec);
  auto device = 0;

  vec.Shard(device);
  ASSERT_EQ(vec.DeviceSize(0), h_vec.size());
  ASSERT_EQ(vec.Size(), h_vec.size());
  auto span = vec.DeviceSpan(0);  // sync to device

  vec.Reshard(-1);  // pull back to cpu.
  ASSERT_EQ(vec.Size(), h_vec.size());
  ASSERT_EQ(vec.DeviceIdx(), -1);

  auto h_vec_1 = vec.HostVector();
  ASSERT_TRUE(std::equal(h_vec_1.cbegin(), h_vec_1.cend(), h_vec.cbegin()));
}

TEST(HostDeviceVector, Reshard) {
  std::vector<int> h_vec (2345);
  for (size_t i = 0; i < h_vec.size(); ++i) {
    h_vec[i] = i;
  }
  HostDeviceVector<int> vec (h_vec);
  auto device = 0;

  vec.Shard(device);
  ASSERT_EQ(vec.DeviceSize(0), h_vec.size());
  ASSERT_EQ(vec.Size(), h_vec.size());
  PlusOne(&vec);

  vec.Reshard(-1);
  ASSERT_EQ(vec.Size(), h_vec.size());
  ASSERT_EQ(vec.DeviceIdx(), -1);

  auto h_vec_1 = vec.HostVector();
  for (size_t i = 0; i < h_vec_1.size(); ++i) {
    ASSERT_EQ(h_vec_1.at(i), i + 1);
  }
}

TEST(HostDeviceVector, Span) {
  HostDeviceVector<float> vec {1.0f, 2.0f, 3.0f, 4.0f};
  vec.Shard(0);
  auto span = vec.DeviceSpan(0);
  ASSERT_EQ(vec.DeviceSize(0), span.size());
  ASSERT_EQ(vec.DevicePointer(0), span.data());
  auto const_span = vec.ConstDeviceSpan(0);
  ASSERT_EQ(vec.DeviceSize(0), span.size());
  ASSERT_EQ(vec.ConstDevicePointer(0), span.data());
}

}  // namespace common
}  // namespace xgboost
