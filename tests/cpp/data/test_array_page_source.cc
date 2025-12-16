/**
 * Copyright 2025, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <memory>  // for make_unique

#include "../../../src/common/linalg_op.h"
#include "../../../src/data/array_page_source.h"
#include "../helpers.h"

namespace xgboost::data {
TEST(ArrayCache, Push) {
  std::size_t n_batches = 4, batch_size = 1024;
  std::size_t shape[2]{n_batches * batch_size, 128};
  auto ctx = MakeCUDACtx(0);
  auto p_cache = std::make_unique<ArrayCache>(&ctx, common::Span<std::size_t>{shape});
  for (std::size_t i = 0; i < n_batches; ++i) {
    auto v = static_cast<float>(i);
    auto page = std::make_shared<ArrayPage>();
    page->gpairs = linalg::Constant<GradientPair>(&ctx, GradientPair{v, v}, batch_size, shape[1]);
    p_cache->Push(std::move(page));
  }
  auto cache = p_cache->Commit();
  for (std::size_t i = 0; i < n_batches; ++i) {
    auto batch = cache.gpairs.Slice(i * batch_size, linalg::All());
    auto e = static_cast<float>(i);
    for (auto v : batch) {
      ASSERT_EQ(v.GetGrad(), e);
    }
  }
}

TEST(ArrayPageSource, Basic) {
  std::map<std::string, std::shared_ptr<Cache>> cache_info;
  std::string cache_prefix = "cache";
  auto id = MakeCache(this, ".ap", false, cache_prefix, &cache_info);
  auto source = std::make_unique<ArrayPageSource>(cache_info.at(id));
}
}  // namespace xgboost::data
