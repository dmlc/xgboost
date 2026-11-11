/**
 * Copyright 2026, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <algorithm>  // for fill
#include <cstddef>    // for size_t
#include <limits>     // for numeric_limits
#include <memory>     // for unique_ptr
#include <string>     // for string
#include <vector>     // for vector

#include "../../../src/common/kernel.h"
#include "../../../src/data/adapter.h"
#include "../../../src/gbm/row_sampler.h"
#include "xgboost/base.h"
#include "xgboost/context.h"
#include "xgboost/gradient.h"
#include "xgboost/task.h"
#include "xgboost/tree_updater.h"

namespace xgboost::gbm {
class RowSamplerTest : public ::testing::TestWithParam<std::string> {};

TEST_P(RowSamplerTest, PairedGradients) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", GetParam()}, {"nthread", "2"}});
  for (auto const* method : {"uniform", "gradient_based"}) {
    tree::TrainParam param;
    param.UpdateAllowUnknown(Args{{"subsample", "0.5"}, {"sampling_method", method}});
    GradientContainer in;
    in.gpair.Reshape(2048, 2);
    in.value_gpair.Reshape(2048, 4);
    auto split = in.gpair.HostView();
    auto value = in.value_gpair.HostView();
    for (std::size_t i = 0; i < split.Shape(0); ++i) {
      for (std::size_t t = 0; t < split.Shape(1); ++t) {
        // A zero Hessian does not mean that a row was excluded by sampling.
        split(i, t) = GradientPair{1.0f + (i % 13) + t, i % 7 == 0 ? 0.0f : 1.0f};
      }
      for (std::size_t t = 0; t < value.Shape(1); ++t) {
        value(i, t) = GradientPair{2.0f, 1.0f};
      }
    }
    auto before_split = in.gpair.Data()->ConstHostVector();
    auto before_value = in.value_gpair.Data()->ConstHostVector();
    GradientContainer out;
    ctx.Rng().seed(17);
    auto sampler = common::DispatchKernel<MakeRowSamplerKernel>(&ctx, param, in);
    sampler->Sample(&ctx, &out);
    EXPECT_EQ(out.gpair.Device(), ctx.Device());
    EXPECT_EQ(out.value_gpair.Device(), ctx.Device());
    EXPECT_EQ(in.gpair.Data()->ConstHostVector(), before_split);
    EXPECT_EQ(in.value_gpair.Data()->ConstHostVector(), before_value);
    auto sampled_split = out.gpair.HostView();
    auto sampled_value = out.value_gpair.HostView();
    std::size_t retained{0};
    std::size_t zero_hessian_retained{0};
    split = in.gpair.HostView();
    for (std::size_t i = 0; i < split.Shape(0); ++i) {
      float weight = sampled_value(i, 0).GetHess();
      retained += weight != 0.0f;
      zero_hessian_retained += weight != 0.0f && split(i, 0).GetHess() == 0.0f;
      for (std::size_t t = 0; t < split.Shape(1); ++t) {
        EXPECT_NEAR(sampled_split(i, t).GetGrad(), split(i, t).GetGrad() * weight, 1.0e-5);
        EXPECT_NEAR(sampled_split(i, t).GetHess(), split(i, t).GetHess() * weight, 1.0e-5);
      }
      for (std::size_t t = 0; t < value.Shape(1); ++t) {
        EXPECT_EQ(sampled_value(i, t).GetGrad(), 2.0f * weight);
        EXPECT_EQ(sampled_value(i, t).GetHess(), weight);
      }
    }
    EXPECT_GT(retained, 800);
    EXPECT_LT(retained, 1250);
    EXPECT_GT(zero_hessian_retained, 0);
    auto first = out.gpair.Data()->ConstHostVector();
    ctx.Rng().seed(17);
    sampler->Sample(&ctx, &out);
    EXPECT_EQ(first, out.gpair.Data()->ConstHostVector());
    sampler->Sample(&ctx, &out);
    EXPECT_NE(first, out.gpair.Data()->ConstHostVector());
  }
}

TEST_P(RowSamplerTest, EmptyAndZeroBudget) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", GetParam()}, {"nthread", "2"}});
  tree::TrainParam param;
  param.UpdateAllowUnknown(Args{{"subsample", "0.1"}, {"sampling_method", "gradient_based"}});
  for (std::size_t n_rows : {0, 2}) {
    GradientContainer in;
    in.gpair.Reshape(n_rows, 2);
    in.value_gpair.Reshape(n_rows, 4);
    std::fill(in.gpair.Data()->HostVector().begin(), in.gpair.Data()->HostVector().end(),
              GradientPair{2.0f, 1.0f});
    std::fill(in.value_gpair.Data()->HostVector().begin(),
              in.value_gpair.Data()->HostVector().end(), GradientPair{3.0f, 1.0f});
    GradientContainer out;
    auto sampler = common::DispatchKernel<MakeRowSamplerKernel>(&ctx, param, in);
    sampler->Sample(&ctx, &out);
    EXPECT_EQ(out.gpair.Shape(), in.gpair.Shape());
    EXPECT_EQ(out.value_gpair.Shape(), in.value_gpair.Shape());
    for (auto g : out.gpair.Data()->ConstHostVector()) {
      EXPECT_EQ(g, GradientPair{});
    }
    for (auto g : out.value_gpair.Data()->ConstHostVector()) {
      EXPECT_EQ(g, GradientPair{});
    }
  }
}

TEST_P(RowSamplerTest, ClearPreviousValueGradient) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", GetParam()}, {"nthread", "2"}});
  tree::TrainParam param;
  param.UpdateAllowUnknown(Args{{"subsample", "1"}});
  GradientContainer in, out;
  in.gpair.Reshape(16, 1);
  in.gpair.Data()->Fill(GradientPair{2.0f, 1.0f});
  out.value_gpair.Reshape(16, 4);
  auto sampler = common::DispatchKernel<MakeRowSamplerKernel>(&ctx, param, in);
  sampler->Sample(&ctx, &out);
  EXPECT_FALSE(out.HasValueGrad());
  EXPECT_EQ(out.gpair.Data()->ConstHostVector(), in.gpair.Data()->ConstHostVector());
}

TEST_P(RowSamplerTest, PositionsForAllRows) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", GetParam()}, {"nthread", "2"}});
  std::vector<float> X(96);
  for (std::size_t i = 0; i < X.size(); ++i) {
    X[i] = i;
  }
  data::DenseAdapter adapter{X.data(), X.size(), 1};
  std::shared_ptr<DMatrix> data{
      DMatrix::Create(&adapter, std::numeric_limits<float>::quiet_NaN(), 2)};
  GradientContainer in, sampled;
  in.gpair.Reshape(X.size(), 1);
  auto grad = in.gpair.HostView();
  for (std::size_t i = 0; i < X.size(); ++i) {
    grad(i, 0) = GradientPair{static_cast<float>(i) - 48.0f, i % 7 == 0 ? 0.0f : 1.0f};
  }
  std::vector<std::string> updaters;
  if (ctx.IsCUDA()) {
    updaters = {"grow_gpu_hist", "grow_gpu_approx"};
  } else if (ctx.IsSycl()) {
    updaters = {"grow_quantile_histmaker_sycl"};
  } else {
    updaters = {"grow_colmaker", "grow_histmaker", "grow_quantile_histmaker"};
  }
  for (auto const& name : updaters) {
    tree::TrainParam param;
    param.UpdateAllowUnknown(
        Args{{"subsample", "0.5"}, {"sampling_method", "gradient_based"}, {"max_depth", "2"}});
    auto sampler = common::DispatchKernel<MakeRowSamplerKernel>(&ctx, param, in);
    sampler->Sample(&ctx, &sampled);
    ObjInfo task{ObjInfo::kRegression};
    std::unique_ptr<TreeUpdater> updater{TreeUpdater::Create(name, &ctx, &task)};
    updater->Configure({});
    RegTree tree;
    std::vector<HostDeviceVector<bst_node_t>> positions(1);
    updater->Update(&param, &sampled, data.get(), common::Span{positions}, {&tree});
    ASSERT_EQ(positions.front().Size(), X.size()) << name;
    for (auto position : positions.front().ConstHostVector()) {
      ASSERT_GE(position, 0) << name;
      ASSERT_TRUE(tree[position].IsLeaf()) << name;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(CPU, RowSamplerTest, ::testing::Values("cpu"));
#if defined(XGBOOST_USE_CUDA)
INSTANTIATE_TEST_SUITE_P(CUDA, RowSamplerTest, ::testing::Values("cuda"));
#endif
#if defined(XGBOOST_USE_SYCL)
INSTANTIATE_TEST_SUITE_P(SYCL, RowSamplerTest, ::testing::Values("sycl"));
#endif
}  // namespace xgboost::gbm
