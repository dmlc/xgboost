/**
 * Copyright 2018-2023 by XGBoost contributors
 */
#include "test_elementwise_metric.h"

namespace xgboost {
namespace metric {
TEST(Metric, DeclareUnifiedTest(RMSE)) { VerifyRMSE(); }

TEST(Metric, DeclareUnifiedTest(RMSLE)) { VerifyRMSLE(); }

TEST(Metric, DeclareUnifiedTest(MAE)) { VerifyMAE(); }

TEST(Metric, DeclareUnifiedTest(MAPE)) { VerifyMAPE(); }

TEST(Metric, DeclareUnifiedTest(MPHE)) { VerifyMPHE(); }

TEST(Metric, DeclareUnifiedTest(LogLoss)) { VerifyLogLoss(); }

TEST(Metric, DeclareUnifiedTest(Error)) { VerifyError(); }

TEST(Metric, DeclareUnifiedTest(PoissonNegLogLik)) { VerifyPoissonNegLogLik(); }

TEST(Metric, DeclareUnifiedTest(MultiRMSE)) { VerifyMultiRMSE(); }

TEST(Metric, DeclareUnifiedTest(Quantile)) { VerifyQuantile(); }

TEST_F(DeclareUnifiedDistributedTest(MetricTest), RMSERowSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyRMSE, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), RMSEColumnSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyRMSE, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), RMSLERowSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyRMSLE, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), RMSLEColumnSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyRMSLE, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MAERowSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyMAE, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MAEColumnSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyMAE, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MAPERowSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyMAPE, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MAPEColumnSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyMAPE, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MPHERowSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyMPHE, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MPHEColumnSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyMPHE, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), LogLossRowSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyLogLoss, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), LogLossColumnSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyLogLoss, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), ErrorRowSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyError, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), ErrorColumnSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyError, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), PoissonNegLogLikRowSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyPoissonNegLogLik, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), PoissonNegLogLikColumnSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyPoissonNegLogLik, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MultiRMSERowSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyMultiRMSE, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MultiRMSEColumnSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyMultiRMSE, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), QuantileRowSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyQuantile, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), QuantileColumnSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyQuantile, DataSplitMode::kCol);
}
}  // namespace metric
}  // namespace xgboost
