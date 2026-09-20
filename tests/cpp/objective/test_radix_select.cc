/**
 * Copyright 2026, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <algorithm>  // std::max, std::min
#include <memory>     // std::unique_ptr
#include <thread>     // std::thread
#include <vector>     // std::vector

#include "../../../src/objective/radix_select.h"
#include "../collective/test_worker.h"   // for TestDistributedGlobal
#include "../helpers.h"                  // for MakeCUDACtx
#include "xgboost/context.h"             // for Context
#include "xgboost/data.h"                // for MetaInfo
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/learner.h"             // for Learner
#include "xgboost/linalg.h"              // for Matrix, Vector
#include "xgboost/objective.h"           // for ObjFunction

namespace xgboost::obj {
namespace {
void TestRadixSelect(Context const* ctx) {
  linalg::Matrix<float> values{
      {-100.0f, 9.0f, 3.0f, -4.0f, 2.0f, 0.0f, 1.0f, 2.0f, 1.0e20f, 1.0f}, {5, 2}, ctx->Device()};
  HostDeviceVector<float> alphas{{0.0f, 0.5f, 1.0f}};
  linalg::Vector<float> out;

  HostDeviceVector<float> weights;
  RadixSelect(ctx, values, weights, alphas, 6, &out);
  auto h_out = out.HostView();
  std::vector<float> expected{-100.0f, 2.0f, 1.0e20f, -4.0f, 1.0f, 9.0f};
  ASSERT_EQ(h_out.Size(), expected.size());
  for (std::size_t i{0}; i < expected.size(); ++i) {
    ASSERT_EQ(h_out(i), expected[i]);
  }

  weights.HostVector() = {0.0f, 1.0f, 2.0f, 4.0f, 8.0f};
  alphas.HostVector() = {0.25f, 0.5f, 0.75f};
  RadixSelect(ctx, values, weights, alphas, 6, &out);
  h_out = out.HostView();
  expected = {1.0f, 1.0e20f, 1.0e20f, 1.0f, 1.0f, 2.0f};
  for (std::size_t i{0}; i < expected.size(); ++i) {
    ASSERT_EQ(h_out(i), expected[i]);
  }

  weights.HostVector().assign(5, 0.0f);
  RadixSelect(ctx, values, weights, alphas, 6, &out);
  h_out = out.HostView();
  for (std::size_t i{0}; i < h_out.Size(); ++i) {
    ASSERT_EQ(h_out(i), 0.0f);
  }
}
}  // namespace

TEST(ObjectiveRadixSelect, Select) {
  Context ctx;
  TestRadixSelect(&ctx);
#if defined(XGBOOST_USE_CUDA)
  ctx = MakeCUDACtx(0);
  TestRadixSelect(&ctx);
#endif  // defined(XGBOOST_USE_CUDA)
}

TEST(ObjectiveRadixSelect, Distributed) {
  auto n_workers =
      static_cast<int>(std::max(1u, std::min(4u, std::thread::hardware_concurrency())));
  collective::TestDistributedGlobal(n_workers, [n_workers] {
    auto rank = collective::GetRank();
    Context ctx;
    collective::GetWorkerLocalThreads(collective::GetWorldSize(), &ctx);
    auto empty = n_workers > 1 && rank == n_workers - 1;
    auto n_rows = empty ? 0 : 2;
    auto n_columns = empty ? 0 : 1;
    linalg::Matrix<float> values({n_rows, n_columns}, ctx.Device());
    if (!empty) {
      auto first = static_cast<float>(2 * rank);
      auto h_values = values.HostView();
      h_values(0, 0) = first;
      h_values(1, 0) = first + 1.0f;
    }
    HostDeviceVector<float> weights;
    HostDeviceVector<float> alphas{{0.25f, 0.5f, 0.75f}};
    linalg::Vector<float> out;
    RadixSelect(&ctx, values, weights, alphas, 3, &out);
    auto h_out = out.HostView();
    auto n_values = 2 * (n_workers > 1 ? n_workers - 1 : 1);
    auto lower = static_cast<float>((n_values + 3) / 4 - 1);
    auto median = static_cast<float>(n_values / 2 - 1);
    auto upper = static_cast<float>((3 * n_values + 3) / 4 - 1);
    ASSERT_EQ(h_out(0), lower);
    ASSERT_EQ(h_out(1), median);
    ASSERT_EQ(h_out(2), upper);
  });
}

TEST(ObjectiveRadixSelect, DistributedAbsoluteError) {
  constexpr bst_target_t n_targets{3};
  auto n_workers =
      static_cast<int>(std::max(1u, std::min(4u, std::thread::hardware_concurrency())));
  collective::TestDistributedGlobal(n_workers, [n_workers, n_targets] {
    auto rank = collective::GetRank();
    Context ctx;
    collective::GetWorkerLocalThreads(collective::GetWorldSize(), &ctx);
    auto empty = n_workers > 1 && rank == n_workers - 1;

    MetaInfo info;
    info.num_row_ = empty ? 0 : 2;
    info.labels.ModifyInplace(
        [&](HostDeviceVector<float>* labels, common::Span<std::size_t> shape) {
          labels->Resize(info.num_row_ * n_targets);
          shape[0] = info.num_row_;
          // The learner supplies the target dimension even on an empty worker.
          shape[1] = n_targets;
          if (!empty) {
            auto first = static_cast<float>(2 * rank);
            labels->HostVector() = {first,        first + 100.0f, first + 200.0f,
                                    first + 1.0f, first + 101.0f, first + 201.0f};
          }
        });

    std::unique_ptr<ObjFunction> objective{ObjFunction::Create("reg:absoluteerror", &ctx)};
    objective->Configure({});
    linalg::Vector<float> base_score;
    objective->InitEstimation(info, &base_score);

    ASSERT_EQ(base_score.Size(), n_targets);
    auto n_values = 2 * (n_workers > 1 ? n_workers - 1 : 1);
    auto expected = static_cast<float>(n_values / 2 - 1);
    for (bst_target_t target{0}; target < n_targets; ++target) {
      ASSERT_EQ(base_score(target), expected + 100.0f * target);
    }

    HostDeviceVector<float> predictions(info.num_row_ * n_targets, 0.0f);
    linalg::Matrix<GradientPair> gpair;
    objective->GetGradient(predictions, info, 0, &gpair);
    ASSERT_EQ(gpair.Shape(1), n_targets);
  });
}

TEST(ObjectiveRadixSelect, DistributedAbsoluteErrorLearner) {
  constexpr bst_target_t n_targets{3};
  constexpr auto n_workers = 2;
  collective::TestDistributedGlobal(n_workers, [n_workers, n_targets] {
    auto rank = collective::GetRank();
    auto empty = n_workers > 1 && rank == n_workers - 1;
    auto Xy =
        RandomDataGenerator{empty ? 0ul : 2ul, 1, 0.0f}.Targets(n_targets).GenerateDMatrix(!empty);

    std::unique_ptr<Learner> learner{Learner::Create({Xy})};
    learner->Configure({{"objective", "reg:absoluteerror"},
                        {"tree_method", "hist"},
                        {"max_depth", "1"},
                        {"min_child_weight", "0"}});
    learner->UpdateOneIter(0, Xy);

    ASSERT_EQ(learner->Groups(), n_targets);
    Json config{Object{}};
    learner->SaveConfig(&config);
    auto base_score = GetBaseScore(config);
    ASSERT_EQ(base_score.size(), n_targets);

    // A new label-less matrix during continuation is normalized by the learner as well.
    auto next =
        RandomDataGenerator{empty ? 0ul : 2ul, 1, 0.0f}.Targets(n_targets).GenerateDMatrix(!empty);
    learner->UpdateOneIter(1, next);
    ASSERT_EQ(next->Info().labels.Shape(1), n_targets);
  });
}

TEST(ObjectiveRadixSelect, DistributedQuantile) {
  constexpr bst_target_t n_targets{3};
  auto n_workers =
      static_cast<int>(std::max(1u, std::min(4u, std::thread::hardware_concurrency())));
  collective::TestDistributedGlobal(n_workers, [n_workers, n_targets] {
    auto rank = collective::GetRank();
    Context ctx;
    collective::GetWorkerLocalThreads(collective::GetWorldSize(), &ctx);
    auto empty = n_workers > 1 && rank == n_workers - 1;

    MetaInfo info;
    info.num_row_ = empty ? 0 : 2;
    info.labels.ModifyInplace(
        [&](HostDeviceVector<float>* labels, common::Span<std::size_t> shape) {
          labels->Resize(info.num_row_);
          shape[0] = info.num_row_;
          shape[1] = empty ? 0 : 1;
          if (!empty) {
            auto first = static_cast<float>(2 * rank);
            labels->HostVector() = {first, first + 1.0f};
          }
        });

    std::unique_ptr<ObjFunction> objective{ObjFunction::Create("reg:quantileerror", &ctx)};
    objective->Configure({{"quantile_alpha", "[0.25, 0.5, 0.75]"}});
    linalg::Vector<float> base_score;
    objective->InitEstimation(info, &base_score);

    ASSERT_EQ(base_score.Size(), n_targets);
    auto n_values = 2 * (n_workers > 1 ? n_workers - 1 : 1);
    ASSERT_EQ(base_score(0), static_cast<float>((n_values + 3) / 4 - 1));
    ASSERT_EQ(base_score(1), static_cast<float>(n_values / 2 - 1));
    ASSERT_EQ(base_score(2), static_cast<float>((3 * n_values + 3) / 4 - 1));

    HostDeviceVector<float> predictions(info.num_row_ * n_targets, 0.0f);
    linalg::Matrix<GradientPair> gpair;
    objective->GetGradient(predictions, info, 0, &gpair);
    ASSERT_EQ(gpair.Shape(1), n_targets);
  });
}
}  // namespace xgboost::obj
