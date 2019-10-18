/*!
 * Copyright 2018-2019 by Contributors
 */
#include <xgboost/linear_updater.h>
#include <xgboost/gbm.h>

#include "../helpers.h"

#include "../../../src/gbm/gblinear_model.h"

TEST(Linear, shotgun) {
  auto mat = xgboost::CreateDMatrix(10, 10, 0);
  auto lparam = xgboost::CreateEmptyGenericParam(GPUIDX);
  {
    auto updater = std::unique_ptr<xgboost::LinearUpdater>(
        xgboost::LinearUpdater::Create("shotgun", &lparam));
    updater->Configure({{"eta", "1."}});
    xgboost::HostDeviceVector<xgboost::GradientPair> gpair(
        (*mat)->Info().num_row_, xgboost::GradientPair(-5, 1.0));
    xgboost::gbm::GBLinearModel model;
    model.param.num_feature = (*mat)->Info().num_col_;
    model.param.num_output_group = 1;
    model.LazyInitModel();
    updater->Update(&gpair, (*mat).get(), &model, gpair.Size());

    ASSERT_EQ(model.bias()[0], 5.0f);

  }
  {
    auto updater = std::unique_ptr<xgboost::LinearUpdater>(
        xgboost::LinearUpdater::Create("shotgun", &lparam));
    EXPECT_ANY_THROW(updater->Configure({{"feature_selector", "random"}}));
  }
  delete mat;
}

TEST(Linear, coordinate) {
  auto mat = xgboost::CreateDMatrix(10, 10, 0);
  auto lparam = xgboost::CreateEmptyGenericParam(GPUIDX);
  auto updater = std::unique_ptr<xgboost::LinearUpdater>(
      xgboost::LinearUpdater::Create("coord_descent", &lparam));
  updater->Configure({{"eta", "1."}});
  xgboost::HostDeviceVector<xgboost::GradientPair> gpair(
      (*mat)->Info().num_row_, xgboost::GradientPair(-5, 1.0));
  xgboost::gbm::GBLinearModel model;
  model.param.num_feature = (*mat)->Info().num_col_;
  model.param.num_output_group = 1;
  model.LazyInitModel();
  updater->Update(&gpair, (*mat).get(), &model, gpair.Size());

  ASSERT_EQ(model.bias()[0], 5.0f);

  delete mat;
}
