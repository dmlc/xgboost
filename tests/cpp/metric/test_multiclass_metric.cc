// Copyright by Contributors
#include "test_multiclass_metric.h"

#include <string>

namespace xgboost {
namespace metric {

TEST(Metric, DeclareUnifiedTest(MultiClassError)) { VerifyMultiClassError(); }

TEST(Metric, DeclareUnifiedTest(MultiClassLogLoss)) { VerifyMultiClassLogLoss(); }

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MultiClassErrorRowSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyMultiClassError, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MultiClassErrorColumnSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyMultiClassError, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MultiClassLogLossRowSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyMultiClassLogLoss, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MultiClassLogLossColumnSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyMultiClassLogLoss, DataSplitMode::kCol);
}
}  // namespace metric
}  // namespace xgboost
