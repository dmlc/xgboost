/**
 * Copyright 2017-2026, XGBoost contributors
 */
#include "test_quantile_obj.h"

#include <xgboost/base.h>        // for Args
#include <xgboost/context.h>     // for Context
#include <xgboost/data.h>        // for MetaInfo
#include <xgboost/objective.h>   // for ObjFunction
#include <xgboost/span.h>        // for Span
#include <xgboost/tree_model.h>  // for RegTree

#include <memory>  // for unique_ptr
#include <vector>  // for vector

#include "../../../src/tree/param.h"      // for TrainParam
#include "../../../src/tree/tree_view.h"  // for MultiTargetTreeView
#include "../helpers.h"                   // CheckConfigReload,MakeCUDACtx,DeclareUnifiedTest
#include "../tree/test_multi_target_tree_model.h"  // for MakeMtTreeForTest
#include "test_objective_helpers.h"  // for MakePositionsForTest, MakeIotaLabelsForTest

namespace xgboost {

void TestQuantile(Context const* ctx) {
  {
    Args args{{"quantile_alpha", "[0.6, 0.8]"}};
    std::unique_ptr<ObjFunction> obj{ObjFunction::Create("reg:quantileerror", ctx)};
    obj->Configure(args);
    CheckConfigReload(obj, "reg:quantileerror");
  }

  Args args{{"quantile_alpha", "0.6"}};
  std::unique_ptr<ObjFunction> obj{ObjFunction::Create("reg:quantileerror", ctx)};
  obj->Configure(args);
  CheckConfigReload(obj, "reg:quantileerror");

  std::vector<float> predts{1.0f, 2.0f, 3.0f};
  std::vector<float> labels{3.0f, 2.0f, 1.0f};
  std::vector<float> weights{1.0f, 1.0f, 1.0f};
  std::vector<float> grad{-0.6f, 0.4f, 0.4f};
  std::vector<float> hess = weights;
  CheckObjFunction(obj, predts, labels, weights, grad, hess);
}

void TestQuantileIntercept(Context const* ctx) {
  Args args{{"quantile_alpha", "[0.6, 0.8]"}};
  std::unique_ptr<ObjFunction> obj{ObjFunction::Create("reg:quantileerror", ctx)};
  obj->Configure(args);

  MetaInfo info;
  info.num_row_ = 10;
  info.labels.ModifyInplace([&](HostDeviceVector<float>* data, common::Span<std::size_t> shape) {
    data->SetDevice(ctx->Device());
    data->Resize(info.num_row_);
    shape[0] = info.num_row_;
    shape[1] = 1;

    auto& h_labels = data->HostVector();
    for (std::size_t i = 0; i < info.num_row_; ++i) {
      h_labels[i] = i;
    }
  });

  linalg::Vector<float> base_scores;
  obj->InitEstimation(info, &base_scores);
  ASSERT_EQ(base_scores.Size(), 2);
  ASSERT_NEAR(base_scores(0), 5.6, kRtEps);
  ASSERT_NEAR(base_scores(1), 7.8, kRtEps);

  for (std::size_t i = 0; i < info.num_row_; ++i) {
    info.weights_.HostVector().emplace_back(info.num_row_ - i - 1.0);
  }

  obj->InitEstimation(info, &base_scores);
  ASSERT_EQ(base_scores.Size(), 2);
  ASSERT_NEAR(base_scores(0), 3.0, kRtEps);
  ASSERT_NEAR(base_scores(1), 5.0, kRtEps);
}

void TestQuantileVectorLeaf(Context const* ctx) {
  Args args{{"quantile_alpha", "[0.25, 0.5, 0.75]"}};
  std::unique_ptr<ObjFunction> obj{ObjFunction::Create("reg:quantileerror", ctx)};
  obj->Configure(args);

  bst_target_t n_targets = 3;
  auto tree = MakeMtTreeForTest(n_targets);

  bst_node_t left_nidx = tree->LeftChild(RegTree::kRoot);
  bst_node_t right_nidx = tree->RightChild(RegTree::kRoot);

  MetaInfo info;
  MakeIotaLabelsForTest(10, 1u, &info);
  HostDeviceVector<bst_node_t> position;
  MakePositionsForTest(info.num_row_, left_nidx, right_nidx, &position);

  HostDeviceVector<float> predt(info.labels.Shape(0) * n_targets, 0.0f);

  tree::TrainParam param;
  param.Init(Args{{"eta", "2.0"}});
  auto lr = param.learning_rate;

  obj->UpdateTreeLeaf(position, info, lr, predt, 0, tree.get());

  auto mt_tree = tree->HostMtView();
  auto left = mt_tree.LeafValue(mt_tree.LeftChild(RegTree::kRoot));
  auto right = mt_tree.LeafValue(mt_tree.RightChild(RegTree::kRoot));
  std::vector<float> sol_left{1.0f, 4.0f, 7.0f};
  std::vector<float> sol_right{11.0f, 14.0f, 17.0f};
  for (std::size_t i = 0; i < left.Size(); ++i) {
    ASSERT_FLOAT_EQ(left(i), sol_left[i]);
    ASSERT_FLOAT_EQ(right(i), sol_right[i]);
  }
}
}  // namespace xgboost
