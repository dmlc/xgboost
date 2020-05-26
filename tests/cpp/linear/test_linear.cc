/*!
 * Copyright 2018-2019 by Contributors
 */
#include <xgboost/linear_updater.h>
#include <xgboost/gbm.h>

#include "../helpers.h"
#include "test_json_io.h"
#include "../../../src/gbm/gblinear_model.h"
#include "xgboost/base.h"

namespace xgboost {

TEST(Linear, Shotgun) {
  size_t constexpr kRows = 10;
  size_t constexpr kCols = 10;

  auto p_fmat = xgboost::RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();

  auto lparam = xgboost::CreateEmptyGenericParam(GPUIDX);
  LearnerModelParam mparam;
  mparam.num_feature = kCols;
  mparam.num_output_group = 1;
  mparam.base_score = 0.5;

  {
    auto updater = std::unique_ptr<xgboost::LinearUpdater>(
        xgboost::LinearUpdater::Create("shotgun", &lparam));
    updater->Configure({{"eta", "1."}});
    xgboost::HostDeviceVector<xgboost::GradientPair> gpair(
        p_fmat->Info().num_row_, xgboost::GradientPair(-5, 1.0));
    xgboost::gbm::GBLinearModel model{&mparam};
    model.LazyInitModel();
    updater->Update(&gpair, p_fmat.get(), &model, gpair.Size());

    ASSERT_EQ(model.Bias()[0], 5.0f);

  }
  {
    auto updater = std::unique_ptr<xgboost::LinearUpdater>(
        xgboost::LinearUpdater::Create("shotgun", &lparam));
    EXPECT_ANY_THROW(updater->Configure({{"feature_selector", "random"}}));
  }
}

TEST(Shotgun, JsonIO) {
  TestUpdaterJsonIO("shotgun");
}

TEST(Linear, coordinate) {
  size_t constexpr kRows = 10;
  size_t constexpr kCols = 10;

  auto p_fmat = xgboost::RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();

  auto lparam = xgboost::CreateEmptyGenericParam(GPUIDX);
  LearnerModelParam mparam;
  mparam.num_feature = kCols;
  mparam.num_output_group = 1;
  mparam.base_score = 0.5;

  auto updater = std::unique_ptr<xgboost::LinearUpdater>(
      xgboost::LinearUpdater::Create("coord_descent", &lparam));
  updater->Configure({{"eta", "1."}});
  xgboost::HostDeviceVector<xgboost::GradientPair> gpair(
      p_fmat->Info().num_row_, xgboost::GradientPair(-5, 1.0));
  xgboost::gbm::GBLinearModel model{&mparam};
  model.LazyInitModel();
  updater->Update(&gpair, p_fmat.get(), &model, gpair.Size());

  ASSERT_EQ(model.Bias()[0], 5.0f);
}

TEST(Coordinate, JsonIO){
  TestUpdaterJsonIO("coord_descent");
}

}  // namespace xgboost
