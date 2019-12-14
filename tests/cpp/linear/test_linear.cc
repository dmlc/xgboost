/*!
 * Copyright 2018-2019 by Contributors
 */
#include <xgboost/linear_updater.h>
#include <xgboost/gbm.h>

#include "../helpers.h"

#include "../../../src/gbm/gblinear_model.h"

namespace xgboost {

TEST(Linear, shotgun) {
  size_t constexpr kRows = 10;
  size_t constexpr kCols = 10;

  auto pp_dmat = xgboost::CreateDMatrix(kRows, kCols, 0);
  auto p_fmat {*pp_dmat};

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

    ASSERT_EQ(model.bias()[0], 5.0f);

  }
  {
    auto updater = std::unique_ptr<xgboost::LinearUpdater>(
        xgboost::LinearUpdater::Create("shotgun", &lparam));
    EXPECT_ANY_THROW(updater->Configure({{"feature_selector", "random"}}));
  }

  delete pp_dmat;
}

TEST(Linear, coordinate) {
  size_t constexpr kRows = 10;
  size_t constexpr kCols = 10;

  auto pp_dmat = xgboost::CreateDMatrix(kRows, kCols, 0);
  auto p_fmat {*pp_dmat};

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

  ASSERT_EQ(model.bias()[0], 5.0f);

  delete pp_dmat;
}

}  // namespace xgboost
