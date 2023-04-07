#include "test_auc.h"

#include <xgboost/metric.h>

namespace xgboost {
namespace metric {

TEST(Metric, DeclareUnifiedTest(BinaryAUC)) { VerifyBinaryAUC(DataSplitMode::kRow); }

TEST(Metric, DeclareUnifiedTest(MultiClassAUC)) { VerifyMultiClassAUC(DataSplitMode::kRow); }

TEST(Metric, DeclareUnifiedTest(RankingAUC)) { VerifyRankingAUC(DataSplitMode::kRow); }

TEST(Metric, DeclareUnifiedTest(PRAUC)) { VerifyPRAUC(DataSplitMode::kRow); }

TEST(Metric, DeclareUnifiedTest(MultiClassPRAUC)) { VerifyMultiClassPRAUC(DataSplitMode::kRow); }

TEST(Metric, DeclareUnifiedTest(RankingPRAUC)) { VerifyRankingPRAUC(DataSplitMode::kRow); }

TEST_F(DeclareUnifiedDistributedTest(MetricTest), BinaryAUCRowSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyBinaryAUC, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), BinaryAUCColumnSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyBinaryAUC, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MultiClassAUCRowSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyMultiClassAUC, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MultiClassAUCColumnSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyMultiClassAUC, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), RankingAUCRowSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyRankingAUC, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), RankingAUCColumnSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyRankingAUC, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), PRAUCRowSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyPRAUC, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), PRAUCColumnSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyPRAUC, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MultiClassPRAUCRowSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyMultiClassPRAUC, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MultiClassPRAUCColumnSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyMultiClassPRAUC, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), RankingPRAUCRowSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyRankingPRAUC, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), RankingPRAUCColumnSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyRankingPRAUC, DataSplitMode::kCol);
}
}  // namespace metric
}  // namespace xgboost
