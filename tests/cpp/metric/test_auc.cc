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
  RunWithInMemoryCommunicator(world_size_, &VerifyBinaryAUC, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), BinaryAUCColumnSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyBinaryAUC, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MultiClassAUCRowSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyMultiClassAUC, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MultiClassAUCColumnSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyMultiClassAUC, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), RankingAUCRowSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyRankingAUC, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), RankingAUCColumnSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyRankingAUC, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), PRAUCRowSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyPRAUC, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), PRAUCColumnSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyPRAUC, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MultiClassPRAUCRowSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyMultiClassPRAUC, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MultiClassPRAUCColumnSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyMultiClassPRAUC, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), RankingPRAUCRowSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyRankingPRAUC, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), RankingPRAUCColumnSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyRankingPRAUC, DataSplitMode::kCol);
}
}  // namespace metric
}  // namespace xgboost
