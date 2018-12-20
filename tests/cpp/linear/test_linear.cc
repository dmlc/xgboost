// Copyright by Contributors
#include <xgboost/linear_updater.h>
#include "../helpers.h"
#include "xgboost/gbm.h"

TEST(Linear, shotgun) {
  auto mat = xgboost::CreateDMatrix(10, 10, 0);
  {
    auto updater = std::unique_ptr<xgboost::LinearUpdater>(
        xgboost::LinearUpdater::Create("shotgun"));
    updater->Init({{"eta", "1."}});
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
        xgboost::LinearUpdater::Create("shotgun"));
    EXPECT_ANY_THROW(updater->Init({{"feature_selector", "random"}}));
  }
  delete mat;
}

TEST(Linear, coordinate) {
  auto mat = xgboost::CreateDMatrix(10, 10, 0);
  auto updater = std::unique_ptr<xgboost::LinearUpdater>(
      xgboost::LinearUpdater::Create("coord_descent"));
  updater->Init({{"eta", "1."}});
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
