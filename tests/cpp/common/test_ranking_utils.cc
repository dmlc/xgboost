/**
 * Copyright 2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>                        // for Test, AssertionResult, Message, TestPartR...
#include <gtest/gtest.h>                        // for ASSERT_NEAR, ASSERT_T...
#include <xgboost/base.h>                       // for Args
#include <xgboost/context.h>                    // for Context
#include <xgboost/string_view.h>                // for StringView

#include <cstdint>                              // for uint32_t
#include <utility>                              // for pair

#include "../../../src/common/ranking_utils.h"  // for LambdaRankParam, ParseMetricName, MakeMet...

namespace xgboost::ltr {
TEST(RankingUtils, LambdaRankParam) {
  // make sure no memory is shared in dmlc parameter.
  LambdaRankParam p0;
  p0.UpdateAllowUnknown(Args{{"lambdarank_num_pair_per_sample", "3"}});
  ASSERT_EQ(p0.NumPair(), 3);

  LambdaRankParam p1;
  p1.UpdateAllowUnknown(Args{{"lambdarank_num_pair_per_sample", "8"}});

  ASSERT_EQ(p0.NumPair(), 3);
  ASSERT_EQ(p1.NumPair(), 8);

  p0.UpdateAllowUnknown(Args{{"lambdarank_num_pair_per_sample", "17"}});
  ASSERT_EQ(p0.NumPair(), 17);
  ASSERT_EQ(p1.NumPair(), 8);
}

TEST(RankingUtils, ParseMetricName) {
  std::uint32_t topn{32};
  bool minus{false};
  auto name = ParseMetricName("ndcg", "3-", &topn, &minus);
  ASSERT_EQ(name, "ndcg@3-");
  ASSERT_EQ(topn, 3);
  ASSERT_TRUE(minus);

  name = ParseMetricName("ndcg", "6", &topn, &minus);
  ASSERT_EQ(topn, 6);
  ASSERT_TRUE(minus);  // unchanged

  minus = false;
  name = ParseMetricName("ndcg", "-", &topn, &minus);
  ASSERT_EQ(topn, 6);  // unchanged
  ASSERT_TRUE(minus);

  name = ParseMetricName("ndcg", nullptr, &topn, &minus);
  ASSERT_EQ(topn, 6);  // unchanged
  ASSERT_TRUE(minus);  // unchanged

  name = ParseMetricName("ndcg", StringView{}, &topn, &minus);
  ASSERT_EQ(topn, 6);  // unchanged
  ASSERT_TRUE(minus);  // unchanged
}

TEST(RankingUtils, MakeMetricName) {
  auto name = MakeMetricName("map", LambdaRankParam::NotSet(), true);
  ASSERT_EQ(name, "map-");
  name = MakeMetricName("map", LambdaRankParam::NotSet(), false);
  ASSERT_EQ(name, "map");
  name = MakeMetricName("map", 2, true);
  ASSERT_EQ(name, "map@2-");
  name = MakeMetricName("map", 2, false);
  ASSERT_EQ(name, "map@2");
}
}  // namespace xgboost::ltr
