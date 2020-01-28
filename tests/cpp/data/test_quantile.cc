// Copyright by Contributors

#include <gtest/gtest.h>
#include "../../../src/common/quantile.h"

namespace xgboost {
namespace common {
TEST(Quantile, Basic) {
  std::vector<float> data = {0, 1, 2, 3, 4};
  WXQuantileSketch<float, float> sketch(data.size(), 1.0 / data.size());
  for (auto x : data) {
    sketch.Push(x);
  }
  WXQuantileSketch<float, float>::SummaryContainer summary;
  sketch.GetSummary(&summary);
  summary.CheckValid(1e-5);
  for (auto x : summary.space) {
    for (auto i = 0ull; i < data.size(); i++) {
      EXPECT_EQ(summary.data[i].value, data[i]);
    }
  }
  using SketchEntry = WXQuantileSketch<float, float>::SummaryContainer::Entry;
  EXPECT_TRUE(std::is_sorted(
      summary.data, summary.data + summary.size,
      [](SketchEntry a, SketchEntry b) { return a.value < b.value; }));
}

TEST(Quantile, Repeats) {
  std::vector<float> data = {0, 1, 0, 1, 0};
  WXQuantileSketch<float, float> sketch(data.size(), 1.0 / data.size());
  for (auto x : data) {
    sketch.Push(x);
  }
  WXQuantileSketch<float, float>::SummaryContainer summary;
  sketch.GetSummary(&summary);
  summary.CheckValid(1e-5);
  EXPECT_EQ(summary.size, 2);
}
};  // namespace common
};  // namespace xgboost
