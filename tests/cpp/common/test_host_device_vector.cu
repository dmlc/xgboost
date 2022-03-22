/*!
 * Copyright 2018 XGBoost contributors
 */

#include <gtest/gtest.h>
#include <thrust/equal.h>
#include <thrust/iterator/counting_iterator.h>

#include "../../../src/common/device_helpers.cuh"
#include <xgboost/host_device_vector.h>

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
  v->SetDevice(device);
  v->Resize(n);

  ASSERT_EQ(v->Size(), n);
  ASSERT_EQ(v->DeviceIdx(), device);
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
  std::copy_n(thrust::make_counting_iterator(0), n, data_h.begin());
}

void PlusOne(HostDeviceVector<int> *v) {
  int device = v->DeviceIdx();
  SetDevice(device);
  thrust::transform(dh::tcbegin(*v), dh::tcend(*v), dh::tbegin(*v),
                    [=]__device__(unsigned int a){ return a + 1; });
  ASSERT_TRUE(v->DeviceCanWrite());
}

void CheckDevice(HostDeviceVector<int>* v,
                 size_t size,
                 unsigned int first,
                 GPUAccess access) {
  ASSERT_EQ(v->Size(), size);
  SetDevice(v->DeviceIdx());

  ASSERT_TRUE(thrust::equal(dh::tcbegin(*v), dh::tcend(*v),
                            thrust::make_counting_iterator(first)));
  ASSERT_TRUE(v->DeviceCanRead());
  // ensure that the device has at most the access specified by access
  ASSERT_EQ(v->DeviceCanWrite(), access == GPUAccess::kWrite);
  ASSERT_EQ(v->HostCanRead(), access == GPUAccess::kRead);
  ASSERT_FALSE(v->HostCanWrite());

  ASSERT_TRUE(thrust::equal(dh::tbegin(*v), dh::tend(*v),
                            thrust::make_counting_iterator(first)));
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

void TestHostDeviceVector(size_t n, int device) {
  HostDeviceVectorSetDeviceHandler hdvec_dev_hndlr(SetDevice);
  HostDeviceVector<int> v;
  InitHostDeviceVector(n, device, &v);
  CheckDevice(&v, n, 0, GPUAccess::kRead);
  PlusOne(&v);
  CheckDevice(&v, n, 1, GPUAccess::kWrite);
  CheckHost(&v, GPUAccess::kRead);
  CheckHost(&v, GPUAccess::kNone);
}

TEST(HostDeviceVector, Basic) {
  size_t n = 1001;
  int device = 0;
  TestHostDeviceVector(n, device);
}

TEST(HostDeviceVector, Copy) {
  size_t n = 1001;
  int device = 0;
  HostDeviceVectorSetDeviceHandler hdvec_dev_hndlr(SetDevice);

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

TEST(HostDeviceVector, SetDevice) {
  std::vector<int> h_vec (2345);
  for (size_t i = 0; i < h_vec.size(); ++i) {
    h_vec[i] = i;
  }
  HostDeviceVector<int> vec (h_vec);
  auto device = 0;

  vec.SetDevice(device);
  ASSERT_EQ(vec.Size(), h_vec.size());
  auto span = vec.DeviceSpan();  // sync to device

  vec.SetDevice(-1);  // pull back to cpu.
  ASSERT_EQ(vec.Size(), h_vec.size());
  ASSERT_EQ(vec.DeviceIdx(), -1);

  auto h_vec_1 = vec.HostVector();
  ASSERT_TRUE(std::equal(h_vec_1.cbegin(), h_vec_1.cend(), h_vec.cbegin()));
}

TEST(HostDeviceVector, Span) {
  HostDeviceVector<float> vec {1.0f, 2.0f, 3.0f, 4.0f};
  vec.SetDevice(0);
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

TEST(HostDeviceVector, Empty) {
  HostDeviceVector<float> vec {1.0f, 2.0f, 3.0f, 4.0f};
  HostDeviceVector<float> another { std::move(vec) };
  ASSERT_FALSE(another.Empty());
  ASSERT_TRUE(vec.Empty());
}

TEST(HostDeviceVector, MGPU_Basic) {  // NOLINT
  if (AllVisibleGPUs() < 2) {
    LOG(WARNING) << "Not testing in multi-gpu environment.";
    return;
  }

  size_t n = 1001;
  int device = 1;
  TestHostDeviceVector(n, device);
}
}  // namespace common
}  // namespace xgboost
