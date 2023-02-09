/**
 * Copyright 2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <cstdint>  // std::uint32_t

#include "../../../src/common/ranking_utils.h"

namespace xgboost {
namespace ltr {
TEST(RankingUtils, MakeMetricName) {
  std::uint32_t topn{32};
  bool minus{false};
  auto name = MakeMetricName("ndcg", "3-", &topn, &minus);
  ASSERT_EQ(name, "ndcg@3-");
  ASSERT_EQ(topn, 3);
  ASSERT_TRUE(minus);

  name = MakeMetricName("ndcg", "6", &topn, &minus);
  ASSERT_EQ(topn, 6);
  ASSERT_TRUE(minus);  // unchanged

  minus = false;
  name = MakeMetricName("ndcg", "-", &topn, &minus);
  ASSERT_EQ(topn, 6);  // unchanged
  ASSERT_TRUE(minus);

  name = MakeMetricName("ndcg", nullptr, &topn, &minus);
  ASSERT_EQ(topn, 6);  // unchanged
  ASSERT_TRUE(minus);  // unchanged

  name = MakeMetricName("ndcg", StringView{}, &topn, &minus);
  ASSERT_EQ(topn, 6);  // unchanged
  ASSERT_TRUE(minus);  // unchanged
}
}  // namespace ltr
}  // namespace xgboost
