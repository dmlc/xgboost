// Copyright by Contributors
#include <xgboost/linear_updater.h>
#include "../helpers.h"
#include "xgboost/gbm.h"

typedef std::pair<std::string, std::string> arg;

TEST(Linear, shotgun) {
  typedef std::pair<std::string, std::string> arg;
  auto mat = CreateDMatrix(10, 10, 0);
  std::vector<bool> enabled(mat->info().num_col, true);
  mat->InitColAccess(enabled, 1.0f, 1 << 16);
  auto updater = std::unique_ptr<xgboost::LinearUpdater>(
      xgboost::LinearUpdater::Create("updater_shotgun"));
  updater->Init({});
  std::vector<xgboost::bst_gpair> gpair(mat->info().num_row,
                                        xgboost::bst_gpair(-5, 1.0));
  xgboost::gbm::GBLinearModel model;
  model.param.num_feature = mat->info().num_col;
  model.param.num_output_group = 1;
  model.LazyInitModel();
  updater->Update(&gpair, mat.get(), &model, gpair.size());

  ASSERT_EQ(model.bias()[0], 5.0f);
}

TEST(Linear, coordinate) {
  typedef std::pair<std::string, std::string> arg;
  auto mat = CreateDMatrix(10, 10, 0);
  std::vector<bool> enabled(mat->info().num_col, true);
  mat->InitColAccess(enabled, 1.0f, 1 << 16);
  auto updater = std::unique_ptr<xgboost::LinearUpdater>(
      xgboost::LinearUpdater::Create("updater_coordinate"));
  updater->Init({});
  std::vector<xgboost::bst_gpair> gpair(mat->info().num_row,
                                        xgboost::bst_gpair(-5, 1.0));
  xgboost::gbm::GBLinearModel model;
  model.param.num_feature = mat->info().num_col;
  model.param.num_output_group = 1;
  model.LazyInitModel();
  updater->Update(&gpair, mat.get(), &model, gpair.size());

  ASSERT_EQ(model.bias()[0], 5.0f);
}