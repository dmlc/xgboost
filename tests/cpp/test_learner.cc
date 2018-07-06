// Copyright by Contributors
#include <gtest/gtest.h>
#include "helpers.h"
#include "xgboost/learner.h"
#include <xgboost/base.h>
#include <xgboost/objective.h>
#include <xgboost/tree_updater.h>

namespace xgboost {
TEST(learner, Test) {
  typedef std::pair<std::string, std::string> arg;
  auto args = {arg("tree_method", "exact")};
  auto dtrain = CreateDMatrix(8, 10, 0);
  auto mat = {dtrain};
  auto learner = std::unique_ptr<Learner>(Learner::Create(mat));
  learner->Configure(args);
}
TEST(learner, UpdateOneIter) {
  typedef std::pair<std::string, std::string> arg;
  auto args = {arg("tree_method", "exact")};
  auto dtrain = CreateDMatrix(8, 2, 0);
  auto mat = {dtrain};
  auto learner = std::unique_ptr<Learner>(Learner::Create(mat));
  learner->Configure(args);
  learner->InitModel();
  dtrain->Info().labels_ = {   1,    1,    0,    0,    0,     1,     0,     1};
  dtrain->Info().weights_ = { 0.5f, 0.52f, 0.71f, 0.73f, -0.5f, -0.47f, -0.28f, -0.26f};
  learner->UpdateOneIter(0, dtrain.get());
}
TEST(learner, UpdaterColMaker) {
  typedef std::pair<std::string, std::string> arg;
  auto args = {arg("tree_method", "exact"),
               arg("num_feature", "2")};
  auto dtrain = CreateDMatrix(8, 2, 0);
  dtrain->Info().labels_ = {   1,    1,    0,    0,    0,     1,     0,     1};
  dtrain->Info().weights_ = { 0.5f, 0.52f, 0.71f, 0.73f, -0.5f, -0.47f, -0.28f, -0.26f};
  dtrain->Info().num_col_ = 2;
  auto ncol = static_cast<int>(dtrain->Info().num_col_);
  std::vector<bool> enabled(ncol, true);
  dtrain->InitColAccess(10, true);
  std::unique_ptr<TreeUpdater> updater(TreeUpdater::Create("grow_colmaker"));
  updater->Init(args);
  std::vector<RegTree*> new_trees;
  std::unique_ptr<RegTree> ptr(new RegTree());
  ptr->param.InitAllowUnknown(args);
  ptr->InitModel();
  new_trees.push_back(ptr.get());
  xgboost::HostDeviceVector<xgboost::GradientPair> out_gpair;
  out_gpair.Resize(8);
  updater->Update(&out_gpair, dtrain.get(), new_trees);
  CHECK_EQ(new_trees[0]->Stat(0).sum_instance, dtrain->Info().num_row_);
  if(new_trees[0]->MaxDepth() > 0) 
    CHECK_EQ(new_trees[0]->Stat(0).sum_instance, new_trees[0]->Stat(1).sum_instance + new_trees[0]->Stat(2).sum_instance);
}
}  // namespace xgboost