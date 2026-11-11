/**
 * Copyright 2020-2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>  // Context
#include <xgboost/span.h>

#include <algorithm>  // for is_sorted, stable_sort
#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t
#include <string>     // for to_string
#include <utility>    // for pair
#include <vector>     // for vector

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

TEST(Algorithm, BlockedStableSort) {
  // The first element is the sort key, the second records the input order of equal keys.
  using Entry = std::pair<std::int32_t, std::size_t>;
  auto by_key = [](Entry const& l, Entry const& r) { return l.first < r.first; };

  for (std::int32_t n_threads : {1, 2, 3, 5, 12}) {
    Context ctx;
    ctx.Init(Args{{"nthread", std::to_string(n_threads)}});
    for (std::size_t n : {0, 1, 8191, 16384, 16385, 100003}) {
      for (std::size_t n_keys : {1, 7, 4096}) {
        std::vector<Entry> inputs(n);
        for (std::size_t i = 0; i < n; ++i) {
          inputs[i] = {static_cast<std::int32_t>((i * 2654435761u) % n_keys), i};
        }
        auto expected = inputs;
        std::stable_sort(expected.begin(), expected.end(), by_key);

        detail::BlockedStableSort(&ctx, inputs.begin(), inputs.end(), by_key);
        // Threads() is capped at the number of cores, so it can be lower than n_threads.
        ASSERT_EQ(inputs, expected) << "threads:" << ctx.Threads() << " n:" << n
                                    << " n_keys:" << n_keys;
      }
    }
  }
}
}  // namespace common
}  // namespace xgboost
