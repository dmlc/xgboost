/**
 * Copyright 2025, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <cstdint>  // for int64_t
#include <tuple>    // for tie

#include "../../../src/common/cuda_rt_utils.h"  // for TotalMemory
#include "../../../src/data/batch_utils.h"      // for AutoHostRatio
#include "../helpers.h"

namespace xgboost::data {
TEST(BatchUtils, CacheHostRatio) {
  {
    bst_idx_t n_cache_bytes = 128;
    double cache_host_ratio = ::xgboost::cuda_impl::AutoHostRatio();
    std::int64_t min_cache_page_bytes = ::xgboost::cuda_impl::AutoCachePageBytes();
    std::tie(cache_host_ratio, min_cache_page_bytes) =
        detail::DftPageSizeHostRatio(n_cache_bytes, false, cache_host_ratio, min_cache_page_bytes);
    ASSERT_EQ(cache_host_ratio, 0.0);  // Assuming the device has more than 256 bytes of memory ..
    ASSERT_GT(min_cache_page_bytes, 0);
    ASSERT_THAT(
        [&] {
          [[maybe_unused]] auto r =
              detail::DftPageSizeHostRatio(n_cache_bytes, false, 2.0, min_cache_page_bytes);
        },
        GMockThrow(R"(cache_host_ratio)"));
  }
  {
    bst_idx_t constexpr kGB = 1024ul * 1024ul * 1024ul;
    bst_idx_t n_cache_bytes = 1024ul * kGB;
    double cache_host_ratio = ::xgboost::cuda_impl::AutoHostRatio();
    std::int64_t min_cache_page_bytes = ::xgboost::cuda_impl::AutoCachePageBytes();
    std::tie(cache_host_ratio, min_cache_page_bytes) =
        detail::DftPageSizeHostRatio(n_cache_bytes, false, cache_host_ratio, min_cache_page_bytes);
    ASSERT_GE(min_cache_page_bytes + 512, curt::TotalMemory() * cuda_impl::CachePageRatio() * 0.5);
    ASSERT_GT(cache_host_ratio, (1.0 - curt::TotalMemory() / static_cast<double_t>(n_cache_bytes)));
    ASSERT_LT(cache_host_ratio, (1.0 - curt::TotalMemory() / (3.0 * n_cache_bytes)));
  }
}
}  // namespace xgboost::data
