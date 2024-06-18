/**
 * Copyright 2018-2023 by XGBoost contributors
 */
#include "test_elementwise_metric.h"

namespace xgboost::metric {
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
  DoTest(VerifyRMSE, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), RMSEColumnSplit) {
  DoTest(VerifyRMSE, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), RMSLERowSplit) {
  DoTest(VerifyRMSLE, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), RMSLEColumnSplit) {
  DoTest(VerifyRMSLE, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MAERowSplit) {
  DoTest(VerifyMAE, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MAEColumnSplit) {
  DoTest(VerifyMAE, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MAPERowSplit) {
  DoTest(VerifyMAPE, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MAPEColumnSplit) {
  DoTest(VerifyMAPE, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MPHERowSplit) {
  DoTest(VerifyMPHE, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MPHEColumnSplit) {
  DoTest(VerifyMPHE, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), LogLossRowSplit) {
  DoTest(VerifyLogLoss, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), LogLossColumnSplit) {
  DoTest(VerifyLogLoss, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), ErrorRowSplit) {
  DoTest(VerifyError, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), ErrorColumnSplit) {
  DoTest(VerifyError, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), PoissonNegLogLikRowSplit) {
  DoTest(VerifyPoissonNegLogLik, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), PoissonNegLogLikColumnSplit) {
  DoTest(VerifyPoissonNegLogLik, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MultiRMSERowSplit) {
  DoTest(VerifyMultiRMSE, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MultiRMSEColumnSplit) {
  DoTest(VerifyMultiRMSE, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), QuantileRowSplit) {
  DoTest(VerifyQuantile, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), QuantileColumnSplit) {
  DoTest(VerifyQuantile, DataSplitMode::kCol);
}
}  // namespace xgboost::metric
