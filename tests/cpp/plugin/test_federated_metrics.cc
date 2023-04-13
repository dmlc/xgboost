/*!
 * Copyright 2023 XGBoost contributors
 */
#include <gtest/gtest.h>

#include "../metric/test_auc.h"
#include "helpers.h"

namespace xgboost {
namespace metric {

class FederatedMetricTest : public BaseFederatedTest {};

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

}  // namespace metric
}  // namespace xgboost
