/**
 * Copyright 2018-2026, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <thrust/equal.h>
#include <thrust/iterator/counting_iterator.h>
#include <xgboost/context.h>
#include <xgboost/host_device_vector.h>

#include "../../../src/common/cuda_rt_utils.h"  // for SetDevice
#include "../../../src/common/device_helpers.cuh"

namespace xgboost::common {
namespace {
void SetDeviceForTest(DeviceOrd device) {
  int n_devices;
  dh::safe_cuda(cudaGetDeviceCount(&n_devices));
  device.ordinal %= n_devices;
  dh::safe_cuda(cudaSetDevice(device.ordinal));
}
}  // namespace

struct HostDeviceVectorSetDeviceHandler {
  template <typename Functor>
  explicit HostDeviceVectorSetDeviceHandler(Functor f) {
    SetCudaSetDeviceHandler(f);
  }

  ~HostDeviceVectorSetDeviceHandler() { SetCudaSetDeviceHandler(nullptr); }
};

void InitHostDeviceVector(size_t n, DeviceOrd device, HostDeviceVector<int>* v,
                          Context const* ctx) {
  v->SetDevice(device, ctx);
  v->Resize(n);

  ASSERT_EQ(v->Size(), n);
  ASSERT_EQ(v->Device(), device);
  ASSERT_TRUE(v->DeviceCanRead());
  ASSERT_TRUE(v->DeviceCanWrite());
  ASSERT_FALSE(v->HostCanRead());
  ASSERT_FALSE(v->HostCanWrite());

  std::vector<int>& data_h = v->HostVector(ctx);
  ASSERT_TRUE(v->HostCanRead());
  ASSERT_TRUE(v->HostCanWrite());
  ASSERT_FALSE(v->DeviceCanRead());
  ASSERT_FALSE(v->DeviceCanWrite());
  ASSERT_EQ(data_h.size(), n);
  std::copy_n(thrust::make_counting_iterator(0), n, data_h.begin());
}

void PlusOne(HostDeviceVector<int>* v) {
  auto device = v->Device();
  SetDeviceForTest(device);
  thrust::transform(dh::tcbegin(*v), dh::tcend(*v), dh::tbegin(*v),
                    [=] __device__(unsigned int a) { return a + 1; });
  ASSERT_TRUE(v->DeviceCanWrite());
}

void CheckDevice(HostDeviceVector<int>* v, size_t size, unsigned int first, GPUAccess access) {
  ASSERT_EQ(v->Size(), size);
  SetDeviceForTest(v->Device());

  ASSERT_TRUE(thrust::equal(dh::tcbegin(*v), dh::tcend(*v), thrust::make_counting_iterator(first)));
  ASSERT_TRUE(v->DeviceCanRead());
  ASSERT_EQ(v->DeviceCanWrite(), access == GPUAccess::kWrite);
  ASSERT_EQ(v->HostCanRead(), access == GPUAccess::kRead);
  ASSERT_FALSE(v->HostCanWrite());

  ASSERT_TRUE(thrust::equal(dh::tbegin(*v), dh::tend(*v), thrust::make_counting_iterator(first)));
  ASSERT_TRUE(v->DeviceCanRead());
  ASSERT_TRUE(v->DeviceCanWrite());
  ASSERT_FALSE(v->HostCanRead());
  ASSERT_FALSE(v->HostCanWrite());
}

void CheckHost(HostDeviceVector<int>* v, GPUAccess access, Context const* ctx) {
  const std::vector<int>& data_h =
      access == GPUAccess::kNone ? v->HostVector(ctx) : v->ConstHostVector(ctx);
  for (size_t i = 0; i < v->Size(); ++i) {
    ASSERT_EQ(data_h.at(i), i + 1);
  }
  ASSERT_TRUE(v->HostCanRead());
  ASSERT_EQ(v->HostCanWrite(), access == GPUAccess::kNone);
  ASSERT_EQ(v->DeviceCanRead(), access == GPUAccess::kRead);
  ASSERT_FALSE(v->DeviceCanWrite());
}

void TestHostDeviceVector(size_t n, Context const* ctx) {
  HostDeviceVectorSetDeviceHandler hdvec_dev_hndlr(curt::SetDevice);
  HostDeviceVector<int> v;
  InitHostDeviceVector(n, ctx->Device(), &v, ctx);
  CheckDevice(&v, n, 0, GPUAccess::kRead);
  PlusOne(&v);
  CheckDevice(&v, n, 1, GPUAccess::kWrite);
  CheckHost(&v, GPUAccess::kRead, ctx);
  CheckHost(&v, GPUAccess::kNone, ctx);
}

TEST(HostDeviceVector, Basic) {
  size_t n = 1001;
  auto ctx = Context{}.MakeCUDA(0);
  TestHostDeviceVector(n, &ctx);
}

TEST(HostDeviceVector, Copy) {
  size_t n = 1001;
  auto ctx = Context{}.MakeCUDA(0);
  HostDeviceVectorSetDeviceHandler hdvec_dev_hndlr(curt::SetDevice);

  HostDeviceVector<int> v;
  {
    HostDeviceVector<int> v1;
    InitHostDeviceVector(n, ctx.Device(), &v1, &ctx);
    v.Resize(v1.Size());
    v.Copy(v1, &ctx);
  }
  CheckDevice(&v, n, 0, GPUAccess::kRead);
  PlusOne(&v);
  CheckDevice(&v, n, 1, GPUAccess::kWrite);
  CheckHost(&v, GPUAccess::kRead, &ctx);
  CheckHost(&v, GPUAccess::kNone, &ctx);
}

TEST(HostDeviceVector, SetDevice) {
  auto ctx = Context{}.MakeCUDA(0);

  std::vector<int> h_vec(2345);
  for (size_t i = 0; i < h_vec.size(); ++i) {
    h_vec[i] = i;
  }
  HostDeviceVector<int> vec(h_vec);

  vec.SetDevice(ctx.Device(), &ctx);
  ASSERT_EQ(vec.Size(), h_vec.size());
  vec.DeviceSpan(&ctx);  // sync to device

  vec.SetDevice(DeviceOrd::CPU(), &ctx);  // pull back to cpu.
  ASSERT_EQ(vec.Size(), h_vec.size());
  ASSERT_EQ(vec.Device(), DeviceOrd::CPU());

  auto h_vec_1 = vec.HostVector(&ctx);
  ASSERT_TRUE(std::equal(h_vec_1.cbegin(), h_vec_1.cend(), h_vec.cbegin()));
}

TEST(HostDeviceVector, Span) {
  auto ctx = Context{}.MakeCUDA(0);

  HostDeviceVector<float> vec{1.0f, 2.0f, 3.0f, 4.0f};
  vec.SetDevice(ctx.Device(), &ctx);
  auto span = vec.DeviceSpan(&ctx);
  ASSERT_EQ(vec.Size(), span.size());
  ASSERT_EQ(vec.DevicePointer(&ctx), span.data());
  auto const_span = vec.ConstDeviceSpan(&ctx);
  ASSERT_EQ(vec.Size(), const_span.size());
  ASSERT_EQ(vec.ConstDevicePointer(&ctx), const_span.data());

  auto h_span = vec.ConstHostSpan(&ctx);
  ASSERT_TRUE(vec.HostCanRead());
  ASSERT_FALSE(vec.HostCanWrite());
  ASSERT_EQ(h_span.size(), vec.Size());
  ASSERT_EQ(h_span.data(), vec.ConstHostPointer(&ctx));

  h_span = vec.HostSpan(&ctx);
  ASSERT_TRUE(vec.HostCanWrite());
}

TEST(HostDeviceVector, Empty) {
  HostDeviceVector<float> vec{1.0f, 2.0f, 3.0f, 4.0f};
  HostDeviceVector<float> another{std::move(vec)};
  ASSERT_FALSE(another.Empty());
  ASSERT_TRUE(vec.Empty());
}

TEST(HostDeviceVector, Resize) {
  auto ctx = Context{}.MakeCUDA(0);

  auto check = [&](HostDeviceVector<float> const& vec) {
    auto const& h_vec = vec.ConstHostSpan(&ctx);
    for (std::size_t i = 0; i < 4; ++i) {
      ASSERT_EQ(h_vec[i], i + 1);
    }
    for (std::size_t i = 4; i < vec.Size(); ++i) {
      ASSERT_EQ(h_vec[i], 3.0);
    }
  };
  {
    HostDeviceVector<float> vec{1.0f, 2.0f, 3.0f, 4.0f};
    vec.SetDevice(ctx.Device(), &ctx);
    vec.ConstDeviceSpan(&ctx);
    ASSERT_TRUE(vec.DeviceCanRead());
    ASSERT_FALSE(vec.DeviceCanWrite());
    vec.DeviceSpan(&ctx);
    vec.Resize(&ctx, 7, 3.0f);
    ASSERT_TRUE(vec.DeviceCanWrite());
    check(vec);
  }
  {
    HostDeviceVector<float> vec{{1.0f, 2.0f, 3.0f, 4.0f}, ctx.Device(), &ctx};
    ASSERT_TRUE(vec.DeviceCanWrite());
    vec.Resize(&ctx, 7, 3.0f);
    ASSERT_TRUE(vec.DeviceCanWrite());
    check(vec);
  }
  {
    HostDeviceVector<float> vec{1.0f, 2.0f, 3.0f, 4.0f};
    ASSERT_TRUE(vec.HostCanWrite());
    vec.Resize(&ctx, 7, 3.0f);
    ASSERT_TRUE(vec.HostCanWrite());
    check(vec);
  }
}
}  // namespace xgboost::common
