/**
 * Copyright 2016-2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>                 // for Test, EXPECT_NEAR, ASSERT_STREQ
#include <xgboost/context.h>             // for Context
#include <xgboost/data.h>                // for MetaInfo, DMatrix
#include <xgboost/linalg.h>              // for Matrix
#include <xgboost/metric.h>              // for Metric

#include <algorithm>                     // for max
#include <memory>                        // for unique_ptr
#include <vector>                        // for vector

#include "test_rank_metric.h"
#include "../helpers.h"                  // for GetMetricEval, CreateEmptyGe...
#include "xgboost/base.h"                // for bst_float, kRtEps
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/json.h"                // for Json, String, Object

namespace xgboost {
namespace metric {

#if !defined(__CUDACC__)
TEST(Metric, AMS) {
  auto ctx = CreateEmptyGenericParam(GPUIDX);
  EXPECT_ANY_THROW(Metric::Create("ams", &ctx));
  Metric* metric = Metric::Create("ams@0.5f", &ctx);
  ASSERT_STREQ(metric->Name(), "ams@0.5");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0.311f, 0.001f);
  EXPECT_NEAR(GetMetricEval(metric,
                            {0.1f, 0.9f, 0.1f, 0.9f},
                            {  0,   0,   1,   1}),
              0.29710f, 0.001f);

  delete metric;
  metric = Metric::Create("ams@0", &ctx);
  ASSERT_STREQ(metric->Name(), "ams@0");
  EXPECT_NEAR(GetMetricEval(metric, {0, 1}, {0, 1}), 0.311f, 0.001f);

  delete metric;
}
#endif

TEST(Metric, DeclareUnifiedTest(Precision)) { VerifyPrecision(); }

TEST(Metric, DeclareUnifiedTest(NDCG)) { VerifyNDCG(); }

TEST(Metric, DeclareUnifiedTest(MAP)) { VerifyMAP(); }

TEST(Metric, DeclareUnifiedTest(NDCGExpGain)) { VerifyNDCGExpGain(); }

TEST_F(DeclareUnifiedDistributedTest(MetricTest), PrecisionRowSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyPrecision, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), PrecisionColumnSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyPrecision, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), NDCGRowSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyNDCG, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), NDCGColumnSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyNDCG, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MAPRowSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyMAP, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), MAPColumnSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyMAP, DataSplitMode::kCol);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), NDCGExpGainRowSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyNDCGExpGain, DataSplitMode::kRow);
}

TEST_F(DeclareUnifiedDistributedTest(MetricTest), NDCGExpGainColumnSplit) {
  RunWithInMemoryCommunicator(world_size_, &VerifyNDCGExpGain, DataSplitMode::kCol);
}
}  // namespace metric
}  // namespace xgboost
