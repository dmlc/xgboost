/**
 * Copyright 2023 by XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/cache.h>
#include <xgboost/data.h>  // for DMatrix

#include <cstddef>         // for size_t
#include <cstdint>         // for uint32_t
#include <thread>          // for thread

#include "helpers.h"       // for RandomDataGenerator

namespace xgboost {
namespace {
struct CacheForTest {
  std::size_t const i;

  explicit CacheForTest(std::size_t k) : i{k} {}
};
}  // namespace

TEST(DMatrixCache, Basic) {
  std::size_t constexpr kRows = 2, kCols = 1, kCacheSize = 4;
  DMatrixCache<CacheForTest> cache{kCacheSize};

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

TEST(DMatrixCache, MultiThread) {
  std::size_t constexpr kRows = 2, kCols = 1, kCacheSize = 3;
  auto p_fmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();

  auto n = std::thread::hardware_concurrency() * 128u;
  CHECK_NE(n, 0);
  std::vector<std::shared_ptr<CacheForTest>> results(n);

  {
    DMatrixCache<CacheForTest> cache{kCacheSize};
    std::vector<std::thread> tasks;
    for (std::uint32_t tidx = 0; tidx < n; ++tidx) {
      tasks.emplace_back([&, i = tidx]() {
        cache.CacheItem(p_fmat, i);

        auto p_fmat_local = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();
        results[i] = cache.CacheItem(p_fmat_local, i);
      });
    }
    for (auto& t : tasks) {
      t.join();
    }
    for (std::uint32_t tidx = 0; tidx < n; ++tidx) {
      ASSERT_EQ(results[tidx]->i, tidx);
    }

    tasks.clear();

    for (std::int32_t tidx = static_cast<std::int32_t>(n - 1); tidx >= 0; --tidx) {
      tasks.emplace_back([&, i = tidx]() {
        cache.CacheItem(p_fmat, i);

        auto p_fmat_local = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();
        results[i] = cache.CacheItem(p_fmat_local, i);
      });
    }
    for (auto& t : tasks) {
      t.join();
    }
    for (std::uint32_t tidx = 0; tidx < n; ++tidx) {
      ASSERT_EQ(results[tidx]->i, tidx);
    }
  }

  {
    DMatrixCache<CacheForTest> cache{n};
    std::vector<std::thread> tasks;
    for (std::uint32_t tidx = 0; tidx < n; ++tidx) {
      tasks.emplace_back([&, tidx]() { results[tidx] = cache.CacheItem(p_fmat, tidx); });
    }
    for (auto& t : tasks) {
      t.join();
    }
    for (std::uint32_t tidx = 0; tidx < n; ++tidx) {
      ASSERT_EQ(results[tidx]->i, tidx);
    }
  }
}
}  // namespace xgboost
