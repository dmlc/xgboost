/**
 * Copyright 2020-2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>  // for Context

#include <algorithm>          // for is_sorted
#include <cstdint>            // for int32_t
#include <vector>             // for vector

#include "../../../src/common/algorithm.h"

namespace xgboost {
namespace common {
TEST(Algorithm, ArgSort) {
  Context ctx;
  std::vector<float> inputs{3.0, 2.0, 1.0};
  auto ret = ArgSort<bst_feature_t>(&ctx, inputs.cbegin(), inputs.cend());
  std::vector<bst_feature_t> sol{2, 1, 0};
  ASSERT_EQ(ret, sol);
}

TEST(Algorithm, Sort) {
  Context ctx;
  ctx.Init(Args{{"nthread", "8"}});
  std::vector<float> inputs{3.0, 1.0, 2.0};

  Sort(&ctx, inputs.begin(), inputs.end(), std::less<>{});
  ASSERT_TRUE(std::is_sorted(inputs.cbegin(), inputs.cend()));

  inputs = {3.0, 1.0, 2.0};
  StableSort(&ctx, inputs.begin(), inputs.end(), std::less<>{});
  ASSERT_TRUE(std::is_sorted(inputs.cbegin(), inputs.cend()));
}

TEST(Algorithm, AllOf) {
  Context ctx;
  auto is_zero = [](auto v) { return v == 0; };

  for (std::size_t n : {3, 16}) {
    std::vector<std::size_t> data(n, 0);
    for (std::int32_t n_threads : {1, 3, 7}) {
      ctx.nthread = n_threads;
      auto ret = AllOf(&ctx, data.cbegin(), data.cend(), is_zero);
      ASSERT_TRUE(ret);
    }

    data[n / 2] = 1;
    for (std::int32_t n_threads : {1, 3, 7}) {
      ctx.nthread = n_threads;
      auto ret = AllOf(&ctx, data.cbegin(), data.cend(), is_zero);
      ASSERT_FALSE(ret);
    }
  }
}

TEST(Algorithm, NoneOf) {
  Context ctx;
  auto is_one = [](auto v) { return v == 1; };

  for (std::size_t n : {3, 16}) {
    std::vector<std::size_t> data(n, 0);
    for (std::int32_t n_threads : {1, 3, 7}) {
      ctx.nthread = n_threads;
      auto ret = NoneOf(&ctx, data.cbegin(), data.cend(), is_one);
      ASSERT_TRUE(ret);
    }

    data[n / 2] = 1;
    for (std::int32_t n_threads : {1, 3, 7}) {
      ctx.nthread = n_threads;
      auto ret = NoneOf(&ctx, data.cbegin(), data.cend(), is_one);
      ASSERT_FALSE(ret);
    }
  }
}
}  // namespace common
}  // namespace xgboost
