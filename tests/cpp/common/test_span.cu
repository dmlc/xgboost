/*!
 * Copyright 2018 XGBoost contributors
 */
#include <gtest/gtest.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include "../../../src/common/device_helpers.cuh"
#include <xgboost/span.h>
#include "test_span.h"

namespace xgboost {
namespace common {

struct TestStatus {
 private:
  int *status_;

 public:
  TestStatus () {
    dh::safe_cuda(cudaMalloc(&status_, sizeof(int)));
    int h_status = 1;
    dh::safe_cuda(cudaMemcpy(status_, &h_status,
                             sizeof(int), cudaMemcpyHostToDevice));
  }
  ~TestStatus() {
    dh::safe_cuda(cudaFree(status_));
  }

  int Get() {
    int h_status;
    dh::safe_cuda(cudaMemcpy(&h_status, status_,
                             sizeof(int), cudaMemcpyDeviceToHost));
    return h_status;
  }

  int* Data() {
    return status_;
  }
};

__global__ void TestFromOtherKernel(Span<float> span) {
  // don't get optimized out
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx >= span.size()) {
    return;
  }
}
// Test converting different T
__global__ void TestFromOtherKernelConst(Span<float const, 16> span) {
  // don't get optimized out
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx >= span.size()) {
    return;
  }
}

/*!
 * \brief Here we just test whether the code compiles.
 */
TEST(GPUSpan, FromOther) {
  thrust::host_vector<float> h_vec (16);
  std::iota(h_vec.begin(), h_vec.end(), 0);

  thrust::device_vector<float> d_vec (h_vec.size());
  thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());
  // dynamic extent
  {
    Span<float> span (d_vec.data().get(), d_vec.size());
    TestFromOtherKernel<<<1, 16>>>(span);
  }
  {
    Span<float> span (d_vec.data().get(), d_vec.size());
    TestFromOtherKernelConst<<<1, 16>>>(span);
  }
  // static extent
  {
    Span<float, 16> span(d_vec.data().get(), d_vec.data().get() + 16);
    TestFromOtherKernel<<<1, 16>>>(span);
  }
  {
    Span<float, 16> span(d_vec.data().get(), d_vec.data().get() + 16);
    TestFromOtherKernelConst<<<1, 16>>>(span);
  }
}

TEST(GPUSpan, Assignment) {
  dh::safe_cuda(cudaSetDevice(0));
  TestStatus status;
  dh::LaunchN(16, TestAssignment{status.Data()});
  ASSERT_EQ(status.Get(), 1);
}

TEST(GPUSpan, TestStatus) {
  dh::safe_cuda(cudaSetDevice(0));
  TestStatus status;
  dh::LaunchN(16, TestTestStatus{status.Data()});
  ASSERT_EQ(status.Get(), -1);
}

template <typename T>
struct TestEqual {
 private:
  T *lhs_, *rhs_;
  int *status_;

 public:
  TestEqual(T* _lhs, T* _rhs, int * _status) :
      lhs_(_lhs), rhs_(_rhs), status_(_status) {}

  XGBOOST_DEVICE void operator()(size_t _idx) {
    bool res = lhs_[_idx] == rhs_[_idx];
    SPAN_ASSERT_TRUE(res, status_);
  }
};

TEST(GPUSpan, WithTrust) {
  dh::safe_cuda(cudaSetDevice(0));
  // Not adviced to initialize span with host_vector, since h_vec.data() is
  // a host function.
  thrust::host_vector<float> h_vec (16);
  std::iota(h_vec.begin(), h_vec.end(), 0);

  thrust::device_vector<float> d_vec (h_vec.size());
  thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());

  // Can't initialize span with device_vector, since d_vec.data() is not raw
  // pointer
  {
    Span<float> s (d_vec.data().get(), d_vec.size());

    ASSERT_EQ(d_vec.size(), s.size());
    ASSERT_EQ(d_vec.data().get(), s.data());
  }

  {
    TestStatus status;
    thrust::device_vector<float> d_vec1 (d_vec.size());
    thrust::copy(thrust::device, d_vec.begin(), d_vec.end(), d_vec1.begin());
    Span<float> s (d_vec1.data().get(), d_vec.size());

    dh::LaunchN(16, TestEqual<float>{
        thrust::raw_pointer_cast(d_vec1.data()),
        s.data(), status.Data()});
    ASSERT_EQ(status.Get(), 1);

    // FIXME(trivialfis): memory error!
    // bool res = thrust::equal(thrust::device,
    //                          d_vec.begin(), d_vec.end(),
    //                          s.begin());
  }
}

TEST(GPUSpan, BeginEnd) {
  dh::safe_cuda(cudaSetDevice(0));
  TestStatus status;
  dh::LaunchN(16, TestBeginEnd{status.Data()});
  ASSERT_EQ(status.Get(), 1);
}

TEST(GPUSpan, RBeginREnd) {
  dh::safe_cuda(cudaSetDevice(0));
  TestStatus status;
  dh::LaunchN(16, TestRBeginREnd{status.Data()});
  ASSERT_EQ(status.Get(), 1);
}

__global__ void TestModifyKernel(Span<float> span) {
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx >= span.size()) {
    return;
  }
  span[idx] = span.size() - idx;
}

TEST(GPUSpan, Modify) {
  thrust::host_vector<float> h_vec (16);
  InitializeRange(h_vec.begin(), h_vec.end());

  thrust::device_vector<float> d_vec (h_vec.size());
  thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());

  Span<float> span (d_vec.data().get(), d_vec.size());

  TestModifyKernel<<<1, 16>>>(span);

  for (size_t i = 0; i < d_vec.size(); ++i) {
    ASSERT_EQ(d_vec[i], d_vec.size() - i);
  }
}

TEST(GPUSpan, Observers) {
  dh::safe_cuda(cudaSetDevice(0));
  TestStatus status;
  dh::LaunchN(16, TestObservers{status.Data()});
  ASSERT_EQ(status.Get(), 1);
}

TEST(GPUSpan, Compare) {
  dh::safe_cuda(cudaSetDevice(0));
  TestStatus status;
  dh::LaunchN(16, TestIterCompare{status.Data()});
  ASSERT_EQ(status.Get(), 1);
}

struct TestElementAccess {
 private:
  Span<float> span_;

 public:
  XGBOOST_DEVICE explicit TestElementAccess (Span<float> _span) : span_(_span) {}

  XGBOOST_DEVICE float operator()(size_t _idx) {
    float tmp = span_[_idx];
    return tmp;
  }
};

TEST(GPUSpanDeathTest, ElementAccess) {
  dh::safe_cuda(cudaSetDevice(0));
  auto test_element_access = []() {
    thrust::host_vector<float> h_vec (16);
    InitializeRange(h_vec.begin(), h_vec.end());

    thrust::device_vector<float> d_vec (h_vec.size());
    thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());

    Span<float> span (d_vec.data().get(), d_vec.size());
    dh::LaunchN(17, TestElementAccess{span});
  };

  testing::internal::CaptureStdout();
  EXPECT_DEATH(test_element_access(), "");
  std::string output = testing::internal::GetCapturedStdout();
}

__global__ void TestFirstDynamicKernel(Span<float> _span) {
  _span.first<static_cast<Span<float>::index_type>(-1)>();
}
__global__ void TestFirstStaticKernel(Span<float> _span) {
  _span.first(static_cast<Span<float>::index_type>(-1));
}
__global__ void TestLastDynamicKernel(Span<float> _span) {
  _span.last<static_cast<Span<float>::index_type>(-1)>();
}
__global__ void TestLastStaticKernel(Span<float> _span) {
  _span.last(static_cast<Span<float>::index_type>(-1));
}

TEST(GPUSpanDeathTest, FirstLast) {
  // We construct vectors multiple times since thrust can not recover from
  // death test.
  auto lambda_first_dy = []() {
    thrust::host_vector<float> h_vec (4);
    InitializeRange(h_vec.begin(), h_vec.end());

    thrust::device_vector<float> d_vec (h_vec.size());
    thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());

    Span<float> span (d_vec.data().get(), d_vec.size());
    TestFirstDynamicKernel<<<1, 1>>>(span);
  };
  testing::internal::CaptureStdout();
  EXPECT_DEATH(lambda_first_dy(), "");
  std::string output = testing::internal::GetCapturedStdout();

  auto lambda_first_static = []() {
    thrust::host_vector<float> h_vec (4);
    InitializeRange(h_vec.begin(), h_vec.end());

    thrust::device_vector<float> d_vec (h_vec.size());
    thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());

    Span<float> span (d_vec.data().get(), d_vec.size());
    TestFirstStaticKernel<<<1, 1>>>(span);
  };
  testing::internal::CaptureStdout();
  EXPECT_DEATH(lambda_first_static(), "");
  output = testing::internal::GetCapturedStdout();

  auto lambda_last_dy = []() {
    thrust::host_vector<float> h_vec (4);
    InitializeRange(h_vec.begin(), h_vec.end());

    thrust::device_vector<float> d_vec (h_vec.size());
    thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());

    Span<float> span (d_vec.data().get(), d_vec.size());
    TestLastDynamicKernel<<<1, 1>>>(span);
  };
  testing::internal::CaptureStdout();
  EXPECT_DEATH(lambda_last_dy(), "");
  output = testing::internal::GetCapturedStdout();

  auto lambda_last_static = []() {
    thrust::host_vector<float> h_vec (4);
    InitializeRange(h_vec.begin(), h_vec.end());

    thrust::device_vector<float> d_vec (h_vec.size());
    thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());

    Span<float> span (d_vec.data().get(), d_vec.size());
    TestLastStaticKernel<<<1, 1>>>(span);
  };
  testing::internal::CaptureStdout();
  EXPECT_DEATH(lambda_last_static(), "");
  output = testing::internal::GetCapturedStdout();
}

namespace {
void TestFrontBack() {
  Span<float> s;
  EXPECT_DEATH(
      {
        // make sure the termination happens inside this test.
        try {
          dh::LaunchN(1, [=] __device__(size_t) { s.front(); });
          dh::safe_cuda(cudaDeviceSynchronize());
          dh::safe_cuda(cudaGetLastError());
        } catch (dmlc::Error const& e) {
          std::terminate();
        }
      },
      "");
  EXPECT_DEATH(
      {
        try {
          dh::LaunchN(1, [=] __device__(size_t) { s.back(); });
          dh::safe_cuda(cudaDeviceSynchronize());
          dh::safe_cuda(cudaGetLastError());
        } catch (dmlc::Error const& e) {
          std::terminate();
        }
      },
      "");
}
}  // namespace

TEST(GPUSpanDeathTest, FrontBack) {
  TestFrontBack();
}

__global__ void TestSubspanDynamicKernel(Span<float> _span) {
  _span.subspan(16, 0);
}
__global__ void TestSubspanStaticKernel(Span<float> _span) {
  _span.subspan<16>();
}
TEST(GPUSpanDeathTest, Subspan) {
  auto lambda_subspan_dynamic = []() {
    thrust::host_vector<float> h_vec (4);
    InitializeRange(h_vec.begin(), h_vec.end());

    thrust::device_vector<float> d_vec (h_vec.size());
    thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());

    Span<float> span (d_vec.data().get(), d_vec.size());
    TestSubspanDynamicKernel<<<1, 1>>>(span);
  };
  testing::internal::CaptureStdout();
  EXPECT_DEATH(lambda_subspan_dynamic(), "");
  std::string output = testing::internal::GetCapturedStdout();

  auto lambda_subspan_static = []() {
    thrust::host_vector<float> h_vec (4);
    InitializeRange(h_vec.begin(), h_vec.end());

    thrust::device_vector<float> d_vec (h_vec.size());
    thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());

    Span<float> span (d_vec.data().get(), d_vec.size());
    TestSubspanStaticKernel<<<1, 1>>>(span);
  };
  testing::internal::CaptureStdout();
  EXPECT_DEATH(lambda_subspan_static(), "");
  output = testing::internal::GetCapturedStdout();
}

TEST(GPUSpanIter, Construct) {
  dh::safe_cuda(cudaSetDevice(0));
  TestStatus status;
  dh::LaunchN(16, TestIterConstruct{status.Data()});
  ASSERT_EQ(status.Get(), 1);
}

TEST(GPUSpanIter, Ref) {
  dh::safe_cuda(cudaSetDevice(0));
  TestStatus status;
  dh::LaunchN(16, TestIterRef{status.Data()});
  ASSERT_EQ(status.Get(), 1);
}

TEST(GPUSpanIter, Calculate) {
  dh::safe_cuda(cudaSetDevice(0));
  TestStatus status;
  dh::LaunchN(16, TestIterCalculate{status.Data()});
  ASSERT_EQ(status.Get(), 1);
}

TEST(GPUSpanIter, Compare) {
  dh::safe_cuda(cudaSetDevice(0));
  TestStatus status;
  dh::LaunchN(16, TestIterCompare{status.Data()});
  ASSERT_EQ(status.Get(), 1);
}

TEST(GPUSpan, AsBytes) {
  dh::safe_cuda(cudaSetDevice(0));
  TestStatus status;
  dh::LaunchN(16, TestAsBytes{status.Data()});
  ASSERT_EQ(status.Get(), 1);
}

TEST(GPUSpan, AsWritableBytes) {
  dh::safe_cuda(cudaSetDevice(0));
  TestStatus status;
  dh::LaunchN(16, TestAsWritableBytes{status.Data()});
  ASSERT_EQ(status.Get(), 1);
}

}  // namespace common
}  // namespace xgboost
