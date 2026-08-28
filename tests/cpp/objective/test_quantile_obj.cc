/**
 * Copyright 2017-2026, XGBoost contributors
 */
#include "test_quantile_obj.h"

#include <xgboost/base.h>       // for Args
#include <xgboost/context.h>    // for Context
#include <xgboost/data.h>       // for MetaInfo
#include <xgboost/objective.h>  // for ObjFunction
#include <xgboost/span.h>       // for Span

#include <algorithm>  // for max
#include <cmath>      // for abs, sqrt, tanh
#include <memory>     // for unique_ptr
#include <numeric>    // for accumulate
#include <vector>     // for vector

#include "../helpers.h"  // CheckConfigReload,MakeCUDACtx,DeclareUnifiedTest

namespace xgboost {
void TestQuantile(Context const* ctx) {
  std::vector<float> const alphas{0.2f, 0.8f};
  Args args{{"quantile_alpha", "[0.2, 0.8]"}};
  std::unique_ptr<ObjFunction> obj{ObjFunction::Create("reg:quantileerror", ctx)};
  obj->Configure(args);
  CheckConfigReload(obj, "reg:quantileerror");
  ASSERT_FALSE(obj->Task().const_hess);

  std::vector<float> const predts{0.0f, 10.0f, 2.0f, 20.0f, 8.0f, 80.0f};
  std::vector<float> const labels{1.0f, 0.0f, 4.0f};
  std::vector<float> const weights{1.0f, 2.0f, 0.5f};

  MetaInfo info;
  info.num_row_ = labels.size();
  info.labels.Reshape(labels.size(), 1);
  info.labels.Data()->HostVector() = labels;
  info.weights_.HostVector() = weights;
  HostDeviceVector<float> predt{predts};

  linalg::Matrix<GradientPair> gpair;
  obj->GetGradient(predt, info, 0, &gpair);
  auto h_gpair = gpair.HostView();
  auto sum_weight = std::accumulate(weights.cbegin(), weights.cend(), 0.0);
  for (std::size_t target{0}; target < alphas.size(); ++target) {
    double root_residual{0.0};
    for (std::size_t row{0}; row < labels.size(); ++row) {
      root_residual +=
          weights[row] * std::sqrt(std::abs(predts[row * alphas.size() + target] - labels[row]));
    }
    auto root_mean = root_residual / sum_weight;
    auto residual_scale = root_mean * root_mean;
    for (std::size_t row{0}; row < labels.size(); ++row) {
      auto residual = predts[row * alphas.size() + target] - labels[row];
      auto x = residual / (0.04 * residual_scale);
      auto tanh_x = std::tanh(x);
      auto ratio = x == 0.0 ? 1.0 : tanh_x / x;
      ratio = std::max(ratio, 3.0e-4);
      auto expected_grad =
          weights[row] * 0.5 * residual_scale * (tanh_x + 1.0 - 2.0 * alphas[target]);
      auto expected_hess = weights[row] * 0.5 / 0.04 * ratio;
      ASSERT_NEAR(h_gpair(row, target).GetGrad(), expected_grad, 1.0e-5);
      ASSERT_NEAR(h_gpair(row, target).GetHess(), expected_hess, 1.0e-5);
    }
  }

  // A tiny-weight extreme residual exercises the relative curvature floor without making the
  // automatic scale large enough to pull the sample back into the smoothing transition.
  Args floor_args{{"quantile_alpha", "0.5"}};
  std::unique_ptr<ObjFunction> floor_obj{ObjFunction::Create("reg:quantileerror", ctx)};
  floor_obj->Configure(floor_args);
  MetaInfo floor_info;
  floor_info.num_row_ = 3;
  floor_info.labels.Reshape(3, 1);
  floor_info.labels.Data()->HostVector() = {0.0f, 0.0f, 0.0f};
  floor_info.weights_.HostVector() = {1.0f, 1.0f, 1.0e-3f};
  HostDeviceVector<float> floor_predt{{0.0f, 0.0f, 1.0f}};
  floor_obj->GetGradient(floor_predt, floor_info, 0, &gpair);
  h_gpair = gpair.HostView();
  ASSERT_NEAR(h_gpair(0, 0).GetHess(), 0.5 / 0.04, 1.0e-5);
  ASSERT_NEAR(h_gpair(2, 0).GetHess(), 1.0e-3 * 0.5 / 0.04 * 3.0e-4, 1.0e-10);

  // Positive weights must not be treated as zero based on their magnitude.
  MetaInfo tiny_weight_info;
  tiny_weight_info.num_row_ = 1;
  tiny_weight_info.labels.Reshape(1, 1);
  tiny_weight_info.labels.Data()->HostVector() = {0.0f};
  tiny_weight_info.weights_.HostVector() = {1.0e-8f};
  HostDeviceVector<float> tiny_weight_predt{{1.0f}};
  floor_obj->GetGradient(tiny_weight_predt, tiny_weight_info, 0, &gpair);
  h_gpair = gpair.HostView();
  ASSERT_GT(std::abs(h_gpair(0, 0).GetGrad()), 0.0f);
  ASSERT_GT(h_gpair(0, 0).GetHess(), 0.0f);

  info.weights_.HostVector() = {0.0f, 0.0f, 0.0f};
  obj->GetGradient(predt, info, 1, &gpair);
  for (auto const& pair : gpair.Data()->HostVector()) {
    ASSERT_EQ(pair.GetGrad(), 0.0f);
    ASSERT_EQ(pair.GetHess(), 0.0f);
  }

  MetaInfo empty;
  empty.labels.Reshape(0, 1);
  HostDeviceVector<float> empty_predt;
  obj->GetGradient(empty_predt, empty, 1, &gpair);
  ASSERT_EQ(gpair.Size(), 0);

  Args transform_args{{"quantile_alpha", "[0.2, 0.5, 0.8]"}};
  std::unique_ptr<ObjFunction> transform{ObjFunction::Create("reg:quantileerror", ctx)};
  transform->Configure(transform_args);
  HostDeviceVector<float> crossing{{0.0f, 2.0f, 1.0f, -1.0f, 3.0f, 2.0f}};
  crossing.SetDevice(ctx->Device());
  transform->PredTransform(&crossing);
  std::vector<float> const expected{0.0f, 1.0f, 2.0f, -1.0f, 2.0f, 3.0f};
  ASSERT_EQ(crossing.HostVector(), expected);
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

  auto check_init = [&] {
    auto n_targets = obj->Targets(info);
    HostDeviceVector<float> zero_predt(info.num_row_ * n_targets, 0.0f, ctx->Device());
    linalg::Matrix<GradientPair> gpair;
    obj->GetGradient(zero_predt, info, 0, &gpair);

    std::vector<float> expected(n_targets);
    auto h_gpair = gpair.HostView();
    for (bst_target_t target{0}; target < n_targets; ++target) {
      double sum_grad{0.0};
      double sum_hess{0.0};
      for (std::size_t row{0}; row < info.num_row_; ++row) {
        sum_grad += h_gpair(row, target).GetGrad();
        sum_hess += h_gpair(row, target).GetHess();
      }
      expected[target] = -sum_grad / std::max(sum_hess, static_cast<double>(kRtEps));
    }
    HostDeviceVector<float> expected_predt{expected};
    expected_predt.SetDevice(ctx->Device());
    obj->PredTransform(&expected_predt);

    linalg::Vector<float> base_scores;
    obj->InitEstimation(info, &base_scores);
    ASSERT_EQ(base_scores.Size(), n_targets);
    auto const& h_expected = expected_predt.HostVector();
    auto h_base_scores = base_scores.HostView();
    for (bst_target_t target{0}; target < n_targets; ++target) {
      ASSERT_NEAR(h_base_scores(target), h_expected[target], 1.0e-5);
    }
  };

  check_init();

  for (std::size_t i = 0; i < info.num_row_; ++i) {
    info.weights_.HostVector().emplace_back(info.num_row_ - i - 1.0);
  }

  check_init();
}
}  // namespace xgboost
