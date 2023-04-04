#include <xgboost/metric.h>
#include "../helpers.h"
#include "test_auc.h"

namespace xgboost {
namespace metric {

TEST(Metric, BinaryAUCRowSplit) {
  auto constexpr kWorldSize = 3;
  RunWithInMemoryCommunicator(kWorldSize, &VerifyBinaryAUC, DataSplitMode::kRow);
}

TEST(Metric, BinaryAUCColumnSplit) {
  auto constexpr kWorldSize = 3;
  RunWithInMemoryCommunicator(kWorldSize, &VerifyBinaryAUC, DataSplitMode::kCol);
}

TEST(Metric, MultiClassAUCRowSplit) {
  auto constexpr kWorldSize = 3;
  RunWithInMemoryCommunicator(kWorldSize, &VerifyMultiClassAUC, DataSplitMode::kRow);
}

TEST(Metric, MultiClassAUCColumnSplit) {
  auto constexpr kWorldSize = 3;
  RunWithInMemoryCommunicator(kWorldSize, &VerifyMultiClassAUC, DataSplitMode::kCol);
}

TEST(Metric, RankingAUCRowSplit) {
  auto constexpr kWorldSize = 3;
  RunWithInMemoryCommunicator(kWorldSize, &VerifyRankingAUC, DataSplitMode::kRow);
}

TEST(Metric, RankingAUCColumnSplit) {
  auto constexpr kWorldSize = 3;
  RunWithInMemoryCommunicator(kWorldSize, &VerifyRankingAUC, DataSplitMode::kCol);
}

TEST(Metric, PRAUCRowSplit) {
  auto constexpr kWorldSize = 3;
  RunWithInMemoryCommunicator(kWorldSize, &VerifyPRAUC, DataSplitMode::kRow);
}

TEST(Metric, PRAUCColumnSplit) {
  auto constexpr kWorldSize = 3;
  RunWithInMemoryCommunicator(kWorldSize, &VerifyPRAUC, DataSplitMode::kCol);
}

TEST(Metric, MultiClassPRAUCRowSplit) {
  auto constexpr kWorldSize = 3;
  RunWithInMemoryCommunicator(kWorldSize, &VerifyMultiClassPRAUC, DataSplitMode::kRow);
}

TEST(Metric, MultiClassPRAUCColumnSplit) {
  auto constexpr kWorldSize = 3;
  RunWithInMemoryCommunicator(kWorldSize, &VerifyMultiClassPRAUC, DataSplitMode::kCol);
}

TEST(Metric, RankingPRAUCRowSplit) {
  auto constexpr kWorldSize = 3;
  RunWithInMemoryCommunicator(kWorldSize, &VerifyRankingPRAUC, DataSplitMode::kRow);
}

TEST(Metric, RankingPRAUCColumnSplit) {
  auto constexpr kWorldSize = 3;
  RunWithInMemoryCommunicator(kWorldSize, &VerifyRankingPRAUC, DataSplitMode::kCol);
}
}  // namespace metric
}  // namespace xgboost
