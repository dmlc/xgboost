// Copyright by Contributors
#include <xgboost/linear_updater.h>
#include <xgboost/gbm.h>

#include "../helpers.h"
#include "../../../src/gbm/gblinear_model.h"

namespace xgboost {

TEST(Linear, GPUCoordinate) {
  size_t constexpr kRows = 10;
  size_t constexpr kCols = 10;

  auto mat = xgboost::CreateDMatrix(kRows, kCols, 0);
  auto lparam = CreateEmptyGenericParam(GPUIDX);

  LearnerModelParam mparam;
  mparam.num_feature = kCols;
  mparam.num_output_group = 1;
  mparam.base_score = 0.5;

  auto updater = std::unique_ptr<xgboost::LinearUpdater>(
      xgboost::LinearUpdater::Create("gpu_coord_descent", &lparam));
  updater->Configure({{"eta", "1."}});
  xgboost::HostDeviceVector<xgboost::GradientPair> gpair(
      (*mat)->Info().num_row_, xgboost::GradientPair(-5, 1.0));
  xgboost::gbm::GBLinearModel model{&mparam};

  model.LazyInitModel();
  updater->Update(&gpair, (*mat).get(), &model, gpair.Size());

  ASSERT_EQ(model.bias()[0], 5.0f);

  delete mat;
}
}  // namespace xgboost