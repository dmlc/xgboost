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

}  // namespace metric
}  // namespace xgboost
