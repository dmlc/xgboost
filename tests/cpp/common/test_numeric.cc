/**
 * Copyright 2022-2026, XGBoost contributors.
 */
#include <gtest/gtest.h>

#include <algorithm>  // for min, max
#include <array>      // for array
#include <cstddef>    // for size_t
#include <limits>     // for numeric_limits
#include <numeric>
#include <stdexcept>  // for runtime_error
#include <vector>     // for vector

#include "../../../src/common/numeric.h"

namespace xgboost::common {
TEST(Numeric, PartialSum) {
  {
    std::vector<size_t> values{1, 2, 3, 4};
    std::vector<size_t> result(values.size() + 1);
    Context ctx;
    PartialSum(ctx.Threads(), values.begin(), values.end(), static_cast<size_t>(0), result.begin());
    std::vector<size_t> sol(values.size() + 1, 0);
    std::partial_sum(values.begin(), values.end(), sol.begin() + 1);
    ASSERT_EQ(sol, result);
  }
  {
    std::vector<double> values{1.5, 2.5, 3.5, 4.5};
    std::vector<double> result(values.size() + 1);
    Context ctx;
    PartialSum(ctx.Threads(), values.begin(), values.end(), 0.0, result.begin());
    std::vector<double> sol(values.size() + 1, 0.0);
    std::partial_sum(values.begin(), values.end(), sol.begin() + 1);
    ASSERT_EQ(sol, result);
  }
}

TEST(Numeric, Reduce) {
  Context ctx;
  ASSERT_TRUE(ctx.IsCPU());
  for (auto n_threads : {1, 4}) {
    ctx.nthread = n_threads;
    for (std::size_t n : {0, 1, 20, 8193}) {
      HostDeviceVector<float> values(n);
      auto& h_values = values.HostVector();
      std::iota(h_values.begin(), h_values.end(), 1.0f);
      ASSERT_EQ(Reduce(&ctx, values), n * (n + 1) / 2);
    }
  }
}

TEST(Numeric, TransformReduce) {
  // Exercise a custom accumulator and combiner with a nonzero identity.
  using Bounds = std::array<int, 2>;
  Bounds identity{std::numeric_limits<int>::max(), std::numeric_limits<int>::lowest()};
  auto reduce = [&](std::size_t size, int n_threads) {
    return TransformReduce(
        size, n_threads, identity,
        [](std::size_t i) -> Bounds {
          auto value = static_cast<int>(i) + 1;
          return {value, value};
        },
        [](Bounds const& a, Bounds const& b) -> Bounds {
          return {std::min(a[0], b[0]), std::max(a[1], b[1])};
        });
  };
  for (auto n_threads : {1, 4}) {
    ASSERT_EQ(reduce(0, n_threads), identity);
    ASSERT_EQ(reduce(8193, n_threads), (Bounds{1, 8193}));
    ASSERT_THROW(
        TransformReduce(8193, n_threads, 0,
                        [](std::size_t) -> int { throw std::runtime_error{"Transform failed"}; }),
        std::runtime_error);
  }
  // Nested serial calls must not index storage with the enclosing OpenMP thread ID.
  ParallelFor(4, 4, [&](auto) { ASSERT_EQ(reduce(8193, 1), (Bounds{1, 8193})); });
}

TEST(Numeric, Iota) {
  Context ctx;
  auto run = [&](std::size_t n) {
    std::vector<float> values(n);
    float init = 1.2f;
    Iota(&ctx, values.begin(), values.end(), init);
    for (std::size_t i = 0; i < values.size(); ++i) {
      ASSERT_EQ(values[i], init + i);
    }
  };
  run(1234);
  run(0);
  run(1);
}
}  // namespace xgboost::common
