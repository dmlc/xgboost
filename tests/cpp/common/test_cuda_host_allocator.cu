/**
 * Copyright 2024-2025, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>  // for Context

#include <vector>

#include "../../../src/common/cuda_pinned_allocator.h"
#include "../../../src/common/cuda_stream.h"       // for DefaultStream
#include "../../../src/common/device_helpers.cuh"
#include "../../../src/common/numeric.h"      // for Iota

namespace xgboost {
TEST(CudaHostMalloc, Pinned) {
  std::vector<float, common::cuda_impl::PinnedAllocator<float>> vec;
  vec.resize(10);
  ASSERT_EQ(vec.size(), 10);
  Context ctx;
  common::Iota(&ctx, vec.begin(), vec.end(), 0);
  float k = 0;
  for (auto v : vec) {
    ASSERT_EQ(v, k);
    ++k;
  }
}

TEST(CudaHostMalloc, Managed) {
  std::vector<float, common::cuda_impl::ManagedAllocator<float>> vec;
  vec.resize(10);
#if defined(__linux__)
#if (CUDA_VERSION / 1000) >= 13
  cudaMemLocation loc;
  loc.type = cudaMemLocationTypeDevice;
  loc.id = 0;
  dh::safe_cuda(
      cudaMemPrefetchAsync(vec.data(), vec.size() * sizeof(float), loc, 0, curt::DefaultStream()));
#else
  dh::safe_cuda(
      cudaMemPrefetchAsync(vec.data(), vec.size() * sizeof(float), 0, curt::DefaultStream()));
#endif  // (CUDA_VERSION / 1000) >= 13
#endif
  curt::DefaultStream().Sync();
}
}  // namespace xgboost
