/**
 * Copyright 2025-2026, XGBoost contributors
 */

#include <gtest/gtest.h>

#include <cstdint>  // for int32_t
#include <set>      // for set
#include "xgboost/span.h"  // for Span
#include "../../../src/common/cuda_stream_pool.h"
#include "../../../src/common/cuda_rt_utils.h"

namespace xgboost::curt {
TEST(RtUtils, Uuid) {
  std::array<unsigned char, kUuidLength> uuid;
  GetUuid(uuid, 0);
  auto str = PrintUuid(uuid);
  ASSERT_EQ(str.substr(0, 4), "GPU-");
  ASSERT_EQ(str.length(), 40);
  ASSERT_EQ(str.size(), str.length());
}

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
