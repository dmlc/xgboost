
// Copyright (c) 2019 by Contributors
#include <gtest/gtest.h>
#include <xgboost/data.h>
#include "../../../src/data/adapter.h"
#include "../../../src/data/device_dmatrix.h"
#include "../helpers.h"
#include <thrust/device_vector.h>
#include "../../../src/data/device_adapter.cuh"
#include "../../../src/gbm/gbtree_model.h"
#include "../common/test_hist_util.h"
#include <xgboost/predictor.h>
using namespace xgboost;  // NOLINT

TEST(DeviceDMatrix, Simple) {
  int num_rows = 10;
  int num_columns = 2;
  auto x = common::GenerateRandom(num_rows, num_columns);
  auto x_device = thrust::device_vector<float>(x);
  auto adapter = common::AdapterFromData(x_device, num_rows, num_columns);

  data::DeviceDMatrix device_dmat(&adapter,
                                  std::numeric_limits<float>::quiet_NaN(), 1);

  auto &batch = *device_dmat.GetBatches<EllpackPage>({0, 256, 0}).begin();

  auto gpu_lparam = CreateEmptyGenericParam(0);

  std::unique_ptr<Predictor> gpu_predictor = std::unique_ptr<Predictor>(
      Predictor::Create("gpu_predictor", &gpu_lparam));

  gpu_predictor->Configure({});
  LearnerModelParam param;
  param.num_output_group = 1;
  gbm::GBTreeModel model = CreateTestModel(&param);
  HostDeviceVector<float> predictions(num_rows);
  PredictionCacheEntry entry;
  gpu_predictor->PredictBatch(&device_dmat, &entry, model, 0);
}
