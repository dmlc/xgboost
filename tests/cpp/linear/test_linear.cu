// Copyright by Contributors
#include <xgboost/linear_updater.h>
#include <xgboost/gbm.h>

#include "../helpers.h"
#include "test_json_io.h"
#include "../../../src/gbm/gblinear_model.h"

namespace xgboost {

TEST(Linear, GPUCoordinate) {
  size_t constexpr kRows = 10;
  size_t constexpr kCols = 10;

  auto mat = xgboost::RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();
  auto ctx = CreateEmptyGenericParam(GPUIDX);

  LearnerModelParam mparam{MakeMP(kCols, .5, 1)};
  auto updater = std::unique_ptr<xgboost::LinearUpdater>(
      xgboost::LinearUpdater::Create("gpu_coord_descent", &ctx));
  updater->Configure({{"eta", "1."}});
  xgboost::HostDeviceVector<xgboost::GradientPair> gpair(
      mat->Info().num_row_, xgboost::GradientPair(-5, 1.0));
  xgboost::gbm::GBLinearModel model{&mparam};

  model.LazyInitModel();
  updater->Update(&gpair, mat.get(), &model, gpair.Size());

  ASSERT_EQ(model.Bias()[0], 5.0f);
}

TEST(GPUCoordinate, JsonIO) {
  TestUpdaterJsonIO("gpu_coord_descent");
}
}  // namespace xgboost
