/**
 * Copyright 2023 by XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/cache.h>
#include <xgboost/data.h>  // DMatrix

#include <cstddef>         // std::size_t

#include "helpers.h"       // RandomDataGenerator

namespace xgboost {
namespace {
struct CacheForTest {
  std::size_t i;

  explicit CacheForTest(std::size_t k) : i{k} {}
};
}  // namespace

TEST(DMatrixCache, Basic) {
  std::size_t constexpr kRows = 2, kCols = 1, kCacheSize = 4;
  DMatrixCache<CacheForTest> cache(kCacheSize);

  auto add_cache = [&]() {
    // Create a lambda function here, so that p_fmat gets deleted upon the
    // end of the lambda. This is to test how the cache handle expired
    // cache entries.
    auto p_fmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();
    cache.CacheItem(p_fmat, 3);
    DMatrix* m = p_fmat.get();
    return m;
  };
  auto m = add_cache();
  ASSERT_EQ(cache.Container().size(), 0);
  ASSERT_THROW(cache.Entry(m), dmlc::Error);

  auto p_fmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();

  auto item = cache.CacheItem(p_fmat, 1);
  ASSERT_EQ(cache.Entry(p_fmat.get())->i, 1);

  std::vector<std::shared_ptr<DMatrix>> items;
  for (std::size_t i = 0; i < kCacheSize * 2; ++i) {
    items.emplace_back(RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix());
    cache.CacheItem(items.back(), i);
    ASSERT_EQ(cache.Entry(items.back().get())->i, i);
    ASSERT_LE(cache.Container().size(), kCacheSize);
    if (i > kCacheSize) {
      auto k = i - kCacheSize - 1;
      ASSERT_THROW(cache.Entry(items[k].get()), dmlc::Error);
    }
  }
}
}  // namespace xgboost
