/**
 * Copyright 2024, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <thread>  // for thread

#include <numeric>                     // for iota
#include <thrust/detail/sequence.inl>  // for sequence

#include "../../../src/common/cuda_rt_utils.h"     // for DrVersion
#include "../../../src/common/device_helpers.cuh"  // for CachingThrustPolicy, PinnedMemory
#include "../../../src/common/device_vector.cuh"
#include "xgboost/global_config.h"  // for GlobalConfigThreadLocalStore
#include "xgboost/windefs.h"  // for xgboost_IS_WIN

namespace dh {
TEST(DeviceUVector, Basic) {
  GlobalMemoryLogger().Clear();
  std::int32_t verbosity{3};
  std::swap(verbosity, xgboost::GlobalConfigThreadLocalStore::Get()->verbosity);
  DeviceUVector<float> uvec;
  uvec.resize(12);
  auto peak = GlobalMemoryLogger().PeakMemory();
  auto n_bytes = sizeof(decltype(uvec)::value_type) * uvec.size();
  ASSERT_EQ(peak, n_bytes);
  std::swap(verbosity, xgboost::GlobalConfigThreadLocalStore::Get()->verbosity);
}

#if defined(__linux__)
namespace {
class TestVirtualMem : public ::testing::TestWithParam<CUmemLocationType> {
 public:
  void Run() {
    auto type = this->GetParam();
    detail::GrowOnlyVirtualMemVec vec{type};
    auto prop = xgboost::cudr::MakeAllocProp(type);
    auto gran = xgboost::cudr::GetAllocGranularity(&prop);
    ASSERT_GE(gran, 2);
    auto data = vec.GetSpan<std::int32_t>(32);  // should be smaller than granularity
    ASSERT_EQ(data.size(), 32);
    static_assert(std::is_same_v<typename decltype(data)::value_type, std::int32_t>);

    std::vector<std::int32_t> h_data(data.size());
    auto check = [&] {
      for (std::size_t i = 0; i < h_data.size(); ++i) {
        ASSERT_EQ(h_data[i], i);
      }
    };
    auto fill = [&](std::int32_t n_orig, xgboost::common::Span<std::int32_t> data) {
      if (type == CU_MEM_LOCATION_TYPE_DEVICE) {
        thrust::sequence(dh::CachingThrustPolicy(), data.data() + n_orig, data.data() + data.size(),
                         n_orig);
        dh::safe_cuda(cudaMemcpy(h_data.data(), data.data(), data.size_bytes(), cudaMemcpyDefault));
      } else {
        std::iota(data.data() + n_orig, data.data() + data.size(), n_orig);
        std::copy_n(data.data(), data.size(), h_data.data());
      }
    };

    fill(0, data);
    check();

    auto n_orig = data.size();
    // Should be smaller than granularity, use already reserved.
    data = vec.GetSpan<std::int32_t>(128);
    h_data.resize(data.size());
    fill(n_orig, data);
    check();
    if (128 < gran) {
      ASSERT_EQ(vec.Capacity(), gran);
    }

    n_orig = data.size();
    data = vec.GetSpan<std::int32_t>(gran / 2);
    h_data.resize(data.size());
    fill(n_orig, data);
    check();
    ASSERT_EQ(vec.Capacity(), gran * 2);

    n_orig = data.size();
    data = vec.GetSpan<std::int32_t>(gran);
    h_data.resize(data.size());
    fill(n_orig, data);
    check();
    ASSERT_EQ(vec.Capacity(), gran * 4);
  }
};
}  // anonymous namespace

TEST_P(TestVirtualMem, Alloc) { this->Run(); }

INSTANTIATE_TEST_SUITE_P(
    Basic, TestVirtualMem,
    ::testing::Values(CU_MEM_LOCATION_TYPE_DEVICE, CU_MEM_LOCATION_TYPE_HOST_NUMA),
    [](::testing::TestParamInfo<TestVirtualMem::ParamType> const& info) -> char const* {
      auto type = info.param;
      switch (type) {
        case CU_MEM_LOCATION_TYPE_DEVICE:
          return "Device";
        case CU_MEM_LOCATION_TYPE_HOST_NUMA:
          return "HostNuma";
        default:
          LOG(FATAL) << "unreachable";
      }
      return nullptr;
    });
#endif  // defined(__linux__)

TEST(TestVirtualMem, Version) {
  std::int32_t major, minor;
  xgboost::curt::DrVersion(&major, &minor);
  LOG(INFO) << "Latest supported CUDA version by the driver:" << major << "." << minor;
  PinnedMemory pinned;
#if defined(xgboost_IS_WIN)
  ASSERT_FALSE(pinned.IsVm());
#else  // defined(xgboost_IS_WIN)
  if (major >= 12 && minor >= 5) {
    ASSERT_TRUE(pinned.IsVm());
  } else {
    ASSERT_FALSE(pinned.IsVm());
  }
#endif  // defined(xgboost_IS_WIN)
}

TEST(AtomitFetch, Max) {
  auto n_threads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads;
  std::atomic<std::int64_t> n{0};
  decltype(n)::value_type add = 64;
  for (decltype(n_threads) t = 0; t < n_threads; ++t) {
    threads.emplace_back([=, &n] {
      for (decltype(add) i = 0; i < add; ++i) {
        detail::AtomicFetchMax(n, static_cast<decltype(add)>(t + i));
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  ASSERT_EQ(n, n_threads - 1 + add - 1);  // 0-based indexing
}
}  // namespace dh
