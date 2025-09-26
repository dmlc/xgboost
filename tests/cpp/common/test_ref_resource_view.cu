/**
 * Copyright 2024-2025, XGBoost Contributors
 */
#if defined(__linux__)

#include <gtest/gtest.h>
#include <thrust/equal.h>                       // for equal
#include <thrust/fill.h>                        // for fill_n
#include <thrust/iterator/constant_iterator.h>  // for make_constant_iterator
#include <thrust/sequence.h>                    // for sequence

#include "../../../src/common/ref_resource_view.cuh"
#include "../../../src/common/threadpool.h"  // for ThreadPool
#include "../helpers.h"                      // for MakeCUDACtx

namespace xgboost::common {
class TestCudaGrowOnly : public ::testing::TestWithParam<std::size_t> {
 public:
  void TestGrow(std::size_t m, std::size_t n) {
    auto ctx = MakeCUDACtx(0);
    ctx.CUDACtx()->Stream().Sync();

    auto ref = MakeCudaGrowOnly<double>(m);
    ASSERT_EQ(ref.size_bytes(), m * sizeof(double));
    thrust::sequence(ctx.CUDACtx()->CTP(), ref.begin(), ref.end(), 0.0);
    auto res = std::dynamic_pointer_cast<common::CudaGrowOnlyResource>(ref.Resource());
    CHECK(res);
    res->Resize(n * sizeof(double));

    auto ref1 = RefResourceView<double>(res->DataAs<double>(), res->Size() / sizeof(double),
                                        ref.Resource());
    ASSERT_EQ(res->Size(), n * sizeof(double));
    ASSERT_EQ(ref1.size(), n);
    thrust::sequence(ctx.CUDACtx()->CTP(), ref1.begin(), ref1.end(), static_cast<double>(0.0));
    std::vector<double> h_vec(ref1.size());
    dh::safe_cuda(cudaMemcpyAsync(h_vec.data(), ref1.data(), ref1.size_bytes(), cudaMemcpyDefault));
    for (std::size_t i = 0; i < h_vec.size(); ++i) {
      ASSERT_EQ(h_vec[i], i);
    }
  }

  void Run(std::size_t n) { this->TestGrow(1024, n); }
};

TEST_P(TestCudaGrowOnly, Resize) { this->Run(this->GetParam()); }

INSTANTIATE_TEST_SUITE_P(RefResourceView, TestCudaGrowOnly, ::testing::Values(1 << 20, 1 << 21));

TEST(HostPinnedMemPool, Alloc) {
  std::vector<RefResourceView<double>> refs;

  {
    // pool goes out of scope before refs does. Test memory safety.
    auto pool = std::make_shared<cuda_impl::HostPinnedMemPool>();
    for (std::size_t i = 0; i < 4; ++i) {
      auto ref = MakeFixedVecWithPinnedMemPool<double>(pool, 128 + i, curt::DefaultStream());
      refs.emplace_back(std::move(ref));
    }
    for (std::size_t i = 0; i < 4; ++i) {
      auto const& ref = refs[i];
      ASSERT_EQ(ref.size(), 128 + i);
      ASSERT_EQ(ref.size_bytes(), ref.size() * sizeof(double));
    }

    // Thread safety.
    auto n_threads = static_cast<std::int32_t>(std::thread::hardware_concurrency());
    common::ThreadPool workers{"tmempool", n_threads, [] {
                               }};
    std::vector<std::future<RefResourceView<double>>> alloc_futs;
    for (std::int32_t i = 0, n = n_threads * 4; i < n; ++i) {
      auto fut = workers.Submit([i, pool] {
        auto ref = MakeFixedVecWithPinnedMemPool<double>(pool, 128 + i, curt::DefaultStream());
        return ref;
      });
      alloc_futs.emplace_back(std::move(fut));
    }
    std::vector<std::future<void>> free_futs(alloc_futs.size());
    for (std::int32_t i = 0, n = n_threads * 4; i < n; ++i) {
      auto fut = workers.Submit([i, pool, &alloc_futs, &free_futs] {
        auto ref = alloc_futs[i].get();
        ASSERT_EQ(ref.size(), 128 + i);
      });
      free_futs[i] = std::move(fut);
    }
    for (std::int32_t i = 0, n = n_threads * 4; i < n; ++i) {
      free_futs[i].get();
    }
  }
}
}  // namespace xgboost::common

#endif  // defined(__linux__)
