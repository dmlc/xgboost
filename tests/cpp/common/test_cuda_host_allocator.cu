/**
 * Copyright 2024, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>  // for Context
#include <xgboost/windefs.h>  // for xgboost_is_WIN

#include <vector>

#include "../../../src/common/cuda_pinned_allocator.h"
#include "../../../src/common/device_helpers.cuh"  // for DefaultStream
#include "../../../src/common/numeric.h"           // for Iota

namespace xgboost {
TEST(CudaHostMalloc, Pinned) {
  std::vector<float, common::cuda_impl::pinned_allocator<float>> vec;
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
  std::vector<float, common::cuda_impl::managed_allocator<float>> vec;
  vec.resize(10);
#if !defined(xgboost_is_WIN)
  dh::safe_cuda(
      cudaMemPrefetchAsync(vec.data(), vec.size() * sizeof(float), 0, dh::DefaultStream()));
#endif
  dh::DefaultStream().Sync();
}
}  // namespace xgboost
