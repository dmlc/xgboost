/*!
 * Copyright 2023 XGBoost contributors
 */
#include <gtest/gtest.h>

#include "../metric/test_auc.h"
#include "../metric/test_elementwise_metric.h"
#include "../metric/test_multiclass_metric.h"
#include "../metric/test_rank_metric.h"
#include "../metric/test_survival_metric.h"
#include "helpers.h"

namespace {
class FederatedMetricTest : public xgboost::BaseFederatedTest {};
}  // anonymous namespace

namespace xgboost {
namespace metric {
TEST_F(FederatedMetricTest, BinaryAUCRowSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyBinaryAUC,
                               DataSplitMode::kRow);
}

TEST_F(FederatedMetricTest, BinaryAUCColumnSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyBinaryAUC,
                               DataSplitMode::kCol);
}

TEST_F(FederatedMetricTest, MultiClassAUCRowSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyMultiClassAUC,
                               DataSplitMode::kRow);
}

TEST_F(FederatedMetricTest, MultiClassAUCColumnSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyMultiClassAUC,
                               DataSplitMode::kCol);
}

TEST_F(FederatedMetricTest, RankingAUCRowSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyRankingAUC,
                               DataSplitMode::kRow);
}

TEST_F(FederatedMetricTest, RankingAUCColumnSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyRankingAUC,
                               DataSplitMode::kCol);
}

TEST_F(FederatedMetricTest, PRAUCRowSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyPRAUC, DataSplitMode::kRow);
}

TEST_F(FederatedMetricTest, PRAUCColumnSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyPRAUC, DataSplitMode::kCol);
}

TEST_F(FederatedMetricTest, MultiClassPRAUCRowSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyMultiClassPRAUC,
                               DataSplitMode::kRow);
}

TEST_F(FederatedMetricTest, MultiClassPRAUCColumnSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyMultiClassPRAUC,
                               DataSplitMode::kCol);
}

TEST_F(FederatedMetricTest, RankingPRAUCRowSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyRankingPRAUC,
                               DataSplitMode::kRow);
}

TEST_F(FederatedMetricTest, RankingPRAUCColumnSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyRankingPRAUC,
                               DataSplitMode::kCol);
}

TEST_F(FederatedMetricTest, RMSERowSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyRMSE, DataSplitMode::kRow);
}

TEST_F(FederatedMetricTest, RMSEColumnSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyRMSE, DataSplitMode::kCol);
}

TEST_F(FederatedMetricTest, RMSLERowSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyRMSLE, DataSplitMode::kRow);
}

TEST_F(FederatedMetricTest, RMSLEColumnSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyRMSLE, DataSplitMode::kCol);
}

TEST_F(FederatedMetricTest, MAERowSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyMAE, DataSplitMode::kRow);
}

TEST_F(FederatedMetricTest, MAEColumnSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyMAE, DataSplitMode::kCol);
}

TEST_F(FederatedMetricTest, MAPERowSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyMAPE, DataSplitMode::kRow);
}

TEST_F(FederatedMetricTest, MAPEColumnSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyMAPE, DataSplitMode::kCol);
}

TEST_F(FederatedMetricTest, MPHERowSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyMPHE, DataSplitMode::kRow);
}

TEST_F(FederatedMetricTest, MPHEColumnSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyMPHE, DataSplitMode::kCol);
}

TEST_F(FederatedMetricTest, LogLossRowSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyLogLoss, DataSplitMode::kRow);
}

TEST_F(FederatedMetricTest, LogLossColumnSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyLogLoss, DataSplitMode::kCol);
}

TEST_F(FederatedMetricTest, ErrorRowSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyError, DataSplitMode::kRow);
}

TEST_F(FederatedMetricTest, ErrorColumnSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyError, DataSplitMode::kCol);
}

TEST_F(FederatedMetricTest, PoissonNegLogLikRowSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyPoissonNegLogLik,
                               DataSplitMode::kRow);
}

TEST_F(FederatedMetricTest, PoissonNegLogLikColumnSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyPoissonNegLogLik,
                               DataSplitMode::kCol);
}

TEST_F(FederatedMetricTest, MultiRMSERowSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyMultiRMSE,
                               DataSplitMode::kRow);
}

TEST_F(FederatedMetricTest, MultiRMSEColumnSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyMultiRMSE,
                               DataSplitMode::kCol);
}

TEST_F(FederatedMetricTest, QuantileRowSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyQuantile,
                               DataSplitMode::kRow);
}

TEST_F(FederatedMetricTest, QuantileColumnSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyQuantile,
                               DataSplitMode::kCol);
}

TEST_F(FederatedMetricTest, MultiClassErrorRowSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyMultiClassError,
                               DataSplitMode::kRow);
}

TEST_F(FederatedMetricTest, MultiClassErrorColumnSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyMultiClassError,
                               DataSplitMode::kCol);
}

TEST_F(FederatedMetricTest, MultiClassLogLossRowSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyMultiClassLogLoss,
                               DataSplitMode::kRow);
}

TEST_F(FederatedMetricTest, MultiClassLogLossColumnSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyMultiClassLogLoss,
                               DataSplitMode::kCol);
}

TEST_F(FederatedMetricTest, PrecisionRowSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyPrecision,
                               DataSplitMode::kRow);
}

TEST_F(FederatedMetricTest, PrecisionColumnSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyPrecision,
                               DataSplitMode::kCol);
}

TEST_F(FederatedMetricTest, NDCGRowSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyNDCG, DataSplitMode::kRow);
}

TEST_F(FederatedMetricTest, NDCGColumnSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyNDCG, DataSplitMode::kCol);
}

TEST_F(FederatedMetricTest, MAPRowSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyMAP, DataSplitMode::kRow);
}

TEST_F(FederatedMetricTest, MAPColumnSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyMAP, DataSplitMode::kCol);
}

TEST_F(FederatedMetricTest, NDCGExpGainRowSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyNDCGExpGain,
                               DataSplitMode::kRow);
}

TEST_F(FederatedMetricTest, NDCGExpGainColumnSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyNDCGExpGain,
                               DataSplitMode::kCol);
}
}  // namespace metric
}  // namespace xgboost

namespace xgboost {
namespace common {
TEST_F(FederatedMetricTest, AFTNegLogLikRowSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyAFTNegLogLik,
                               DataSplitMode::kRow);
}

TEST_F(FederatedMetricTest, AFTNegLogLikColumnSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyAFTNegLogLik,
                               DataSplitMode::kCol);
}

TEST_F(FederatedMetricTest, IntervalRegressionAccuracyRowSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyIntervalRegressionAccuracy,
                               DataSplitMode::kRow);
}

TEST_F(FederatedMetricTest, IntervalRegressionAccuracyColumnSplit) {
  RunWithFederatedCommunicator(kWorldSize, server_->Address(), &VerifyIntervalRegressionAccuracy,
                               DataSplitMode::kCol);
}
}  // namespace common
}  // namespace xgboost
