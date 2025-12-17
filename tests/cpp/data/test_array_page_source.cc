/**
 * Copyright 2025, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <memory>  // for make_unique

#include "../../../src/common/linalg_op.h"
#include "../../../src/data/array_page_source.h"
#include "../helpers.h"

namespace xgboost::data {
namespace {
auto CreateCache(std::size_t n_batches, std::size_t batch_size) {
  std::size_t shape[2]{n_batches * batch_size, 128};
  auto ctx = MakeCUDACtx(0);
  auto p_cache = std::make_unique<ArrayCacheWriter>(&ctx, common::Span<std::size_t>{shape});
  for (std::size_t i = 0; i < n_batches; ++i) {
    auto v = static_cast<float>(i);
    auto page = std::make_shared<ArrayPage>();
    page->gpairs = linalg::Constant<GradientPair>(&ctx, GradientPair{v, v}, batch_size, shape[1]);
    p_cache->Push(std::move(page));
  }
  return p_cache->Commit();
}
}  // namespace

TEST(ArrayCache, Push) {
  std::size_t n_batches = 4, batch_size = 1024;
  auto cache = CreateCache(n_batches, batch_size);
  for (std::size_t i = 0; i < n_batches; ++i) {
    auto batch = cache->gpairs.Slice(i * batch_size, linalg::All());
    auto e = static_cast<float>(i);
    for (auto v : batch) {
      ASSERT_EQ(v.GetGrad(), e);
    }
  }
}

TEST(ArrayPageSource, Basic) {
  std::map<std::string, std::shared_ptr<Cache>> cache_info;
  std::string cache_prefix = "grad";
  auto id = MakeCache(this, ".ap", true, cache_prefix, &cache_info);

  std::size_t n_batches = 4, batch_size = 1024;
  auto cache = CreateCache(n_batches, batch_size);

  std::vector<bst_idx_t> batch_ptr{0};
  for (std::size_t i = 0; i < n_batches; ++i) {
    batch_ptr.push_back(batch_size + batch_ptr[i]);
  }
  ASSERT_EQ(batch_ptr, cache->batch_ptr);

  auto n_targets = cache->gpairs.Shape(1);
  auto source = std::make_shared<ArrayPageSource>(std::move(cache), std::move(batch_ptr), n_targets,
                                                  cache_info.at(id));
  auto batch_set = BatchSet{BatchIterator<ArrayPage>{source}};
  std::int32_t k = 0;
  for (auto const& page : batch_set) {
    ASSERT_EQ(page.gpairs.Shape(0), batch_size);
    ASSERT_EQ(page.gpairs.Shape(1), n_targets);
    std::cout << page.gpairs(0, 0) << ", ";
    for (auto v : page.gpairs.HostView()) {
      ASSERT_EQ(static_cast<float>(k), v.GetGrad());
      ASSERT_EQ(static_cast<float>(k), v.GetHess());
    }
    ++k;
  }
}
}  // namespace xgboost::data
