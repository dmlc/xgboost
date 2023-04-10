// Copyright by Contributors
#include "test_multiclass_metric.h"

#include <string>

namespace xgboost {
namespace metric {

TEST(Metric, DeclareUnifiedTest(MultiClassError)) { VerifyMultiClassError(); }

TEST(Metric, DeclareUnifiedTest(MultiClassLogLoss)) { VerifyMultiClassLogLoss(); }

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MultiClassErrorRowSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyMultiClassError, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MultiClassErrorColumnSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyMultiClassError, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MultiClassLogLossRowSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyMultiClassLogLoss, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MultiClassLogLossColumnSplit) {
  RunWithInMemoryCommunicator(n_gpus_, &VerifyMultiClassLogLoss, DataSplitMode::kCol);
}
}  // namespace metric
}  // namespace xgboost
