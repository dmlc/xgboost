/**
 * Copyright 2025, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <thrust/sequence.h>  // for sequence

#include <cstddef>  // for size_t
#include <cstdint>  // for uint8_t
#include <memory>   // for make_shared
#include <tuple>    // for tuple

#include "../../../src/common/cuda_context.cuh"         // for CUDAContext
#include "../../../src/common/cuda_pinned_allocator.h"  // for HostPinnedMemPool
#include "../../../src/common/device_compression.cuh"
#include "../../../src/common/device_helpers.cuh"     // for ToSpan
#include "../../../src/common/device_vector.cuh"      // for DeviceUVector
#include "../../../src/common/ref_resource_view.cuh"  // for MakeFixedVecWithPinnedMemPool
#include "../helpers.h"                               // for MakeCUDACtx

namespace xgboost::dc {
// We skip the tests but keep the code at compilation time if nvcomp is not enabled. This
// helps us to ensure correct symbol definitions.
TEST(NvComp, Snappy) {
#if !defined(XGBOOST_USE_NVCOMP)
  GTEST_SKIP_("XGBoost is not compiled with nvcomp.");
#endif
  auto ctx = MakeCUDACtx(0);
  auto cuctx = ctx.CUDACtx();
  dh::DeviceUVector<common::CompressedByteT> in(1024);
  thrust::sequence(ctx.CUDACtx()->CTP(), in.begin(), in.end(), 0);
  dh::DeviceUVector<std::uint8_t> compr;

  std::size_t chunk_size = 512;
  auto params = CompressSnappy(&ctx, dh::ToSpan(in), &compr, chunk_size);
  ASSERT_GE(params.size(), 1);

  auto pool = std::make_shared<common::cuda_impl::HostPinnedMemPool>();
  auto h_in =
      common::MakeFixedVecWithPinnedMemPool<std::uint8_t>(pool, compr.size(), cuctx->Stream());
  dh::safe_cuda(cudaMemcpyAsync(h_in.data(), compr.data(), compr.size() * sizeof(std::uint8_t),
                                cudaMemcpyDefault, cuctx->Stream()));

  dh::device_vector<common::CompressedByteT> dout(in.size(), 0);
  auto mgr = MakeSnappyDecomprMgr(cuctx->Stream(), pool, params, h_in.ToSpan());
  DecompressSnappy(cuctx->Stream(), mgr, dh::ToSpan(dout), true);

  bool eq = thrust::equal(ctx.CUDACtx()->CTP(), dout.cbegin(), dout.cend(), in.cbegin());
  ASSERT_TRUE(eq);

  auto const& status = GetGlobalDeStatus();
  ASSERT_LT(status.max_output_size, 1ul << 24);
}

class TestNvComp : public ::testing::TestWithParam<std::tuple<std::size_t, std::size_t>> {
 public:
  void Run(std::size_t n_bytes, std::size_t n_chunk_bytes) {
    auto ctx = MakeCUDACtx(0);
    auto cuctx = ctx.CUDACtx();

    dh::DeviceUVector<common::CompressedByteT> in(n_bytes);
    thrust::sequence(ctx.CUDACtx()->CTP(), in.begin(), in.end(), 0);
    dh::DeviceUVector<std::uint8_t> compr;

    auto params = CompressSnappy(&ctx, dh::ToSpan(in), &compr, n_chunk_bytes);
    if (n_bytes != 0) {
      ASSERT_GE(params.size(), 1);
    } else {
      ASSERT_TRUE(params.empty());
    }
    if (n_chunk_bytes < n_bytes) {
      ASSERT_GE(params.size(), n_bytes / n_chunk_bytes);
    }

    auto pool = std::make_shared<common::cuda_impl::HostPinnedMemPool>();

    CuMemParams out_params;
    auto page = CoalesceCompressedBuffersToHost(cuctx->Stream(), pool, params, compr, &out_params);

    dh::device_vector<common::CompressedByteT> dout(in.size(), 0);
    auto mgr = MakeSnappyDecomprMgr(cuctx->Stream(), pool, out_params, page.ToSpan());
    DecompressSnappy(cuctx->Stream(), mgr, dh::ToSpan(dout), true);

    bool eq = thrust::equal(ctx.CUDACtx()->CTP(), dout.cbegin(), dout.cend(), in.cbegin());
    ASSERT_TRUE(eq);
  }
};

TEST_P(TestNvComp, HostBuf) {
#if !defined(XGBOOST_USE_NVCOMP)
  GTEST_SKIP_("XGBoost is not compiled with nvcomp.");
#endif
  auto [n_bytes, n_chunk_bytes] = this->GetParam();
  this->Run(n_bytes, n_chunk_bytes);
}

INSTANTIATE_TEST_SUITE_P(TestNvComp, TestNvComp,
                         ::testing::Combine(::testing::Values(0, 1, 512, 1024),
                                            ::testing::Values(1, 256, 512, 1024, 2048)));
}  // namespace xgboost::dc
