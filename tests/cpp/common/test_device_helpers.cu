
/*!
 * Copyright 2017 XGBoost contributors
 */
#include <thrust/device_vector.h>
#include <xgboost/base.h>
#include "../../../src/common/device_helpers.cuh"
#include "../helpers.h"
#include "gtest/gtest.h"

TEST(SumReduce, Test) {
  thrust::device_vector<float> data(100, 1.0f);
  auto sum = dh::SumReduction(data.data().get(), data.size());
  ASSERT_NEAR(sum, 100.0f, 1e-5);
}

