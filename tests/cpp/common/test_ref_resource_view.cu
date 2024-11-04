/**
 * Copyright 2024, XGBoost Contributors
 */
#if defined(__linux__)

#include <gtest/gtest.h>
#include <thrust/equal.h>                       // for equal
#include <thrust/fill.h>                        // for fill_n
#include <thrust/iterator/constant_iterator.h>  // for make_constant_iterator
#include <thrust/sequence.h>                    // for sequence

#include "../../../src/common/ref_resource_view.cuh"
#include "../helpers.h"  // for MakeCUDACtx

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
}  // namespace xgboost::common

#endif  // defined(__linux__)
