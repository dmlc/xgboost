// Copyright by Contributors
#include "test_multiclass_metric.h"

#include <string>

namespace xgboost {
namespace metric {

TEST(Metric, DeclareUnifiedTest(MultiClassError)) { VerifyMultiClassError(); }

TEST(Metric, DeclareUnifiedTest(MultiClassLogLoss)) { VerifyMultiClassLogLoss(); }

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MultiClassErrorRowSplit) {
  DoTest(VerifyMultiClassError, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MultiClassErrorColumnSplit) {
  DoTest(VerifyMultiClassError, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MultiClassLogLossRowSplit) {
  DoTest(VerifyMultiClassLogLoss, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MultiClassLogLossColumnSplit) {
  DoTest(VerifyMultiClassLogLoss, DataSplitMode::kCol);
}
}  // namespace metric
}  // namespace xgboost
