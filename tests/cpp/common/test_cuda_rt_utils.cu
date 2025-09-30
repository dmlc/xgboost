/**
 * Copyright 2025, XGBoost contributors
 */

#include <gtest/gtest.h>

#include <cstdint>  // for int32_t
#include <set>      // for set

#include "../../../src/common/cuda_stream_pool.h"

namespace xgboost::curt {
TEST(RtUtils, StreamPool) {
  auto n_streams = 16;
  auto pool = std::make_unique<StreamPool>(n_streams);
  std::set<cudaStream_t> hdls;

  for (std::int32_t i = 0; i < n_streams; ++i) {
    hdls.insert(cudaStream_t{pool->Next()});
  }

  ASSERT_EQ(hdls.size(), n_streams);
  ASSERT_EQ(hdls.size(), pool->Size());

  for (std::int32_t i = 0; i < n_streams; ++i) {
    hdls.insert(cudaStream_t{pool->Next()});
  }
  ASSERT_EQ(hdls.size(), n_streams);
  ASSERT_EQ(hdls.size(), pool->Size());
}
}  // namespace xgboost::curt
