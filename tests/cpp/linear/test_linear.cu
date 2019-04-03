// Copyright by Contributors
#include <xgboost/linear_updater.h>
#include "../helpers.h"
#include "xgboost/gbm.h"

namespace xgboost {

TEST(Linear, GPUCoordinate) {
  dh::safe_cuda(cudaSetDevice(0));
  auto mat = xgboost::CreateDMatrix(10, 10, 0);
  auto updater = std::unique_ptr<xgboost::LinearUpdater>(
      xgboost::LinearUpdater::Create("gpu_coord_descent"));
  updater->Init({{"eta", "1."}, {"n_gpus", "1"}});
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

#if defined(XGBOOST_USE_NCCL)
TEST(Linear, MGPU_GPUCoordinate) {
  dh::safe_cuda(cudaSetDevice(0));
  {
    auto mat = xgboost::CreateDMatrix(10, 10, 0);
    auto updater = std::unique_ptr<xgboost::LinearUpdater>(
        xgboost::LinearUpdater::Create("gpu_coord_descent"));
    updater->Init({{"eta", "1."}, {"n_gpus", "-1"}});
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

  dh::safe_cuda(cudaSetDevice(0));
  {
    auto mat = xgboost::CreateDMatrix(10, 10, 0);
    auto updater = std::unique_ptr<xgboost::LinearUpdater>(
        xgboost::LinearUpdater::Create("gpu_coord_descent"));
    updater->Init({
        {"eta", "1."},
        {"n_gpus", "-1"},
        {"gpu_id", "1"}});
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
}
#endif

}  // namespace xgboost