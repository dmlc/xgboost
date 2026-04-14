/**
 * Copyright 2022-2026, XGBoost contributors.
 */
#include <gtest/gtest.h>

#include <numeric>

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
  HostDeviceVector<float> values(20);
  auto& h_values = values.HostVector();
  std::iota(h_values.begin(), h_values.end(), 0.0f);
  auto sum = Reduce(&ctx, values);
  ASSERT_EQ(sum, (values.Size() - 1) * values.Size() / 2);
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
