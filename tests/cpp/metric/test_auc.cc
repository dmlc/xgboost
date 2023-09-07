#include "test_auc.h"

#include <xgboost/metric.h>

namespace xgboost {
namespace metric {

TEST(Metric, DeclareUnifiedTest(BinaryAUC)) { VerifyBinaryAUC(); }

TEST(Metric, DeclareUnifiedTest(MultiClassAUC)) { VerifyMultiClassAUC(); }

TEST(Metric, DeclareUnifiedTest(RankingAUC)) { VerifyRankingAUC(); }

TEST(Metric, DeclareUnifiedTest(PRAUC)) { VerifyPRAUC(); }

TEST(Metric, DeclareUnifiedTest(MultiClassPRAUC)) { VerifyMultiClassPRAUC(); }

TEST(Metric, DeclareUnifiedTest(RankingPRAUC)) { VerifyRankingPRAUC(); }

TEST_F(DeclareUnifiedDistributedTest(MetricTest), BinaryAUCRowSplit) {
  DoTest(VerifyBinaryAUC, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), BinaryAUCColumnSplit) {
  DoTest(VerifyBinaryAUC, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MultiClassAUCRowSplit) {
  DoTest(VerifyMultiClassAUC, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MultiClassAUCColumnSplit) {
  DoTest(VerifyMultiClassAUC, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), RankingAUCRowSplit) {
  DoTest(VerifyRankingAUC, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), RankingAUCColumnSplit) {
  DoTest(VerifyRankingAUC, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), PRAUCRowSplit) {
  DoTest(VerifyPRAUC, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), PRAUCColumnSplit) {
  DoTest(VerifyPRAUC, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MultiClassPRAUCRowSplit) {
  DoTest(VerifyMultiClassPRAUC, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MultiClassPRAUCColumnSplit) {
  DoTest(VerifyMultiClassPRAUC, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), RankingPRAUCRowSplit) {
  DoTest(VerifyRankingPRAUC, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), RankingPRAUCColumnSplit) {
  DoTest(VerifyRankingPRAUC, DataSplitMode::kCol);
}
}  // namespace metric
}  // namespace xgboost
