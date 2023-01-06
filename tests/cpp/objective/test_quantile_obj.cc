/**
 * Copyright 2023 by XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>       // Args
#include <xgboost/context.h>    // Context
#include <xgboost/objective.h>  // ObjFunction
#include <xgboost/span.h>       // Span

#include <memory>               // std::unique_ptr
#include <vector>               // std::vector

#include "../helpers.h"         // CheckConfigReload,CreateEmptyGenericParam,DeclareUnifiedTest

namespace xgboost {
TEST(Objective, DeclareUnifiedTest(Quantile)) {
  Context ctx = CreateEmptyGenericParam(GPUIDX);

  {
    Args args{{"quantile_alpha", "[0.6, 0.8]"}};
    std::unique_ptr<ObjFunction> obj{ObjFunction::Create("reg:quantileerror", &ctx)};
    obj->Configure(args);
    CheckConfigReload(obj, "reg:quantileerror");
  }

  Args args{{"quantile_alpha", "0.6"}};
  std::unique_ptr<ObjFunction> obj{ObjFunction::Create("reg:quantileerror", &ctx)};
  obj->Configure(args);
  CheckConfigReload(obj, "reg:quantileerror");

  std::vector<float> predts{1.0f, 2.0f, 3.0f};
  std::vector<float> labels{3.0f, 2.0f, 1.0f};
  std::vector<float> weights{1.0f, 1.0f, 1.0f};
  std::vector<float> grad{-0.6f, 0.4f, 0.4f};
  std::vector<float> hess = weights;
  CheckObjFunction(obj, predts, labels, weights, grad, hess);
}

TEST(Objective, DeclareUnifiedTest(QuantileIntercept)) {
  Context ctx = CreateEmptyGenericParam(GPUIDX);
  Args args{{"quantile_alpha", "[0.6, 0.8]"}};
  std::unique_ptr<ObjFunction> obj{ObjFunction::Create("reg:quantileerror", &ctx)};
  obj->Configure(args);

  MetaInfo info;
  info.num_row_ = 10;
  info.labels.ModifyInplace([&](HostDeviceVector<float>* data, common::Span<std::size_t> shape) {
    data->SetDevice(ctx.gpu_id);
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
  ASSERT_EQ(base_scores.Size(), 1) << "Vector is not yet supported.";
  // mean([5.6, 7.8])
  ASSERT_NEAR(base_scores(0), 6.7, kRtEps);

  for (std::size_t i = 0; i < info.num_row_; ++i) {
    info.weights_.HostVector().emplace_back(info.num_row_ - i - 1.0);
  }

  obj->InitEstimation(info, &base_scores);
  ASSERT_EQ(base_scores.Size(), 1) << "Vector is not yet supported.";
  // mean([3, 5])
  ASSERT_NEAR(base_scores(0), 4.0, kRtEps);
}
}  // namespace xgboost
