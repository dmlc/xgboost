#include <gtest/gtest.h>
#include "../../../src/common/ranking_utils.h"

namespace xgboost {
TEST(RankingUtils, IDCG) {
  std::vector<uint32_t> scores{2, 2, 1, 0};
  float IDCG = CalcInvIDCG(scores, scores.size());
  ASSERT_FLOAT_EQ(IDCG, 1.0f / 5.39279f);
  float ndcg = CalcNDCGAtK(scores, scores, scores.size());
  ASSERT_EQ(ndcg, 1);
}
}  // namespace xgboost
