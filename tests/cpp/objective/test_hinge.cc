/**
 * Copyright 2018-2023, XGBoost Contributors
 */
#include <xgboost/objective.h>
#include <xgboost/context.h>
#include <limits>

#include "../helpers.h"
#include "test_hinge.h"
#include "../../../src/common/linalg_op.h"
namespace xgboost {

void TestHingeObj(const Context* ctx) {
  std::unique_ptr<ObjFunction> obj{ObjFunction::Create("binary:hinge", ctx)};

  float eps = std::numeric_limits<xgboost::bst_float>::min();
  std::vector<float> predt{-1.0f, -0.5f, 0.5f, 1.0f, -1.0f, -0.5f, 0.5f, 1.0f};
  std::vector<float> label{ 0.0f,  0.0f, 0.0f, 0.0f,  1.0f,  1.0f,  1.0f, 1.0f};
  std::vector<float> grad{0.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 0.0f};
  std::vector<float> hess{eps, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, eps};

  CheckObjFunction(obj, predt, label, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, grad, hess);
  CheckObjFunction(obj, predt, label, {/* Empty weight. */}, grad, hess);

  ASSERT_EQ(obj->DefaultEvalMetric(), StringView{"error"});

  MetaInfo info;
  info.num_row_ = label.size();
  info.labels.Reshape(info.num_row_, 3);
  ASSERT_EQ(obj->Targets(info), 3);
  auto h_labels = info.labels.HostView();
  for (std::size_t j = 0; j < obj->Targets(info); ++j) {
    for (std::size_t i = 0; i < info.num_row_; ++i) {
      h_labels(i, j) = label[i];
    }
  }
  linalg::Tensor<float, 2> t_predt{};
  t_predt.Reshape(info.labels.Shape());
  for (std::size_t j = 0; j < obj->Targets(info); ++j) {
    for (std::size_t i = 0; i < info.num_row_; ++i) {
      t_predt(i, j) = predt[i];
    }
  }
  linalg::Matrix<GradientPair> out_gpair;
  obj->GetGradient(*t_predt.Data(), info, 0, &out_gpair);

  for (std::size_t j = 0; j < obj->Targets(info); ++j) {
    auto gh = out_gpair.Slice(linalg::All(), j);
    ASSERT_EQ(gh.Size(), info.num_row_);
    for (std::size_t i = 0; i < gh.Size(); ++i) {
      ASSERT_EQ(gh(i).GetGrad(), grad[i]);
      ASSERT_EQ(gh(i).GetHess(), hess[i]);
    }
  }
}
}  // namespace xgboost
