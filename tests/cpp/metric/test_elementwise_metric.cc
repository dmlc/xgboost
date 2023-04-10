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
  RunWithInMemoryCommunicator(n_gpus_, &VerifyRMSE, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), RMSEColumnSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyRMSE, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), RMSLERowSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyRMSLE, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), RMSLEColumnSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyRMSLE, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MAERowSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyMAE, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MAEColumnSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyMAE, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MAPERowSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyMAPE, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MAPEColumnSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyMAPE, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MPHERowSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyMPHE, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MPHEColumnSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyMPHE, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), LogLossRowSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyLogLoss, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), LogLossColumnSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyLogLoss, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), ErrorRowSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyError, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), ErrorColumnSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyError, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), PoissonNegLogLikRowSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyPoissonNegLogLik, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), PoissonNegLogLikColumnSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyPoissonNegLogLik, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MultiRMSERowSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyMultiRMSE, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MultiRMSEColumnSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyMultiRMSE, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), QuantileRowSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyQuantile, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), QuantileColumnSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyQuantile, DataSplitMode::kCol);
}
}  // namespace metric
}  // namespace xgboost
