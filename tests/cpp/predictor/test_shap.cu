/**
 * Copyright 2017-2026, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/json.h>
#include <xgboost/learner.h>

#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "../../../src/gbm/gbtree_model.h"
#include "../../../src/predictor/interpretability/shap.h"
#include "../helpers.h"
#include "test_predictor.h"
#include "test_shap.h"
#include "xgboost/linalg.h"
#include "xgboost/tree_model.h"

namespace xgboost {

namespace {
void CheckShapUnusedFeatures(Context const* ctx) {
  bst_feature_t constexpr kFeatures = 128;
  bst_target_t constexpr kGroups = 2;
  bst_idx_t constexpr kRows = 5;
  // Noncontiguous split indices, with different active features in each output group.
  std::array<std::array<bst_feature_t, 2>, kGroups> const features{{{3, 127}, {42, 126}}};
  for (bool constant : {false, true}) {
    std::size_t shape[1]{kGroups};
    linalg::Vector<float> base_score{shape, ctx->Device()};
    base_score.Data()->HostVector() = {0.125f, 0.25f};
    std::as_const(base_score).HostView();
    if (ctx->Device().IsCUDA()) {
      std::as_const(base_score).View(ctx->Device());
    }
    LearnerModelState state{kFeatures, std::move(base_score), kGroups, kGroups,
                            MultiStrategy::kOneOutputPerTree};
    gbm::GBTreeModel model{&state, ctx};
    for (bst_target_t g = 0; g < kGroups; ++g) {
      auto tree = std::make_unique<RegTree>();
      float value = static_cast<float>(g + 1);
      if (constant) {
        (*tree)[RegTree::kRoot].SetLeaf(value);
        tree->Stat(RegTree::kRoot).sum_hess = 4.0f;
      } else {
        tree->Expand({{RegTree::kRoot, features[g][0], 0.5f, true},
                      {0.0f, 4.0f},
                      {0.0f, 2.0f},
                      {0.0f, 2.0f},
                      0.0f});
        auto right = (*tree)[RegTree::kRoot].RightChild();
        tree->Expand(
            {{right, features[g][1], 0.5f, true}, {0.0f, 2.0f}, {0.0f, 1.0f}, {value, 1.0f}, 0.0f});
        tree->FinalizeLeaves(1.0f);
      }
      gbm::TreesOneGroup trees;
      trees.emplace_back(std::move(tree));
      model.CommitModelGroup(std::move(trees), g);
    }
    std::vector<float> data(kRows * kFeatures, 0.0f);
    for (bst_idx_t r = 0; r < kRows; ++r) {
      for (auto const& indices : features) {
        data[r * kFeatures + indices[0]] = static_cast<float>(r % 2);
        data[r * kFeatures + indices[1]] = static_cast<float>((r / 2) % 2);
      }
    }
    auto dmat = GetDMatrixFromData(data, kRows, kFeatures);
    for (bool has_margin : {false, true}) {
      if (has_margin) {
        dmat->Info().base_margin_.Reshape(kRows, kGroups);
        auto& margin = dmat->Info().base_margin_.Data()->HostVector();
        for (bst_idx_t i = 0; i < kRows * kGroups; ++i) {
          margin[i] = static_cast<float>(i) / 8.0f;
        }
      }
      for (bst_tree_t tree_end : {1, 2}) {
        std::vector<float> weights{0.5f, 0.25f};
        HostDeviceVector<float> output;
        interpretability::ShapInteractionValues(ctx, dmat.get(), &output, model, tree_end, &weights,
                                                false);
        auto const& actual = output.ConstHostVector();
        auto constexpr kCols = kFeatures + 1;
        ASSERT_EQ(actual.size(), kRows * kGroups * kCols * kCols);
        for (bst_idx_t r = 0; r < kRows; ++r) {
          for (bst_target_t g = 0; g < kGroups; ++g) {
            auto offset = (r * kGroups + g) * kCols * kCols;
            std::vector<float> expected(kCols * kCols, 0.0f);
            float scale = g < tree_end ? static_cast<float>(g + 1) * weights[g] : 0.0f;
            float bias = has_margin ? static_cast<float>(r * kGroups + g) / 8.0f
                                    : static_cast<float>(g + 1) / 8.0f;
            expected[kFeatures * kCols + kFeatures] = bias + (constant ? scale : 0.25f * scale);
            if (!constant) {
              auto [i, j] = features[g];
              float vi = 0.5f * static_cast<float>(r % 2);
              float vj = 0.5f * static_cast<float>((r / 2) % 2);
              float full = static_cast<float>((r % 2) && ((r / 2) % 2));
              float pair = 0.5f * (full - vi - vj + 0.25f) * scale;
              expected[i * kCols + j] = expected[j * kCols + i] = pair;
              expected[i * kCols + i] = 0.5f * (vi - 0.25f + full - vj) * scale - pair;
              expected[j * kCols + j] = 0.5f * (vj - 0.25f + full - vi) * scale - pair;
            }
            for (std::size_t i = 0; i < expected.size(); ++i) {
              EXPECT_NEAR(actual[offset + i], expected[i], 2e-6f)
                  << "constant=" << constant << ", margin=" << has_margin
                  << ", tree_end=" << tree_end << ", row=" << r << ", group=" << g;
            }
          }
        }
      }
    }
  }
}

}  // namespace

TEST(GPUPredictor, ShapUnusedFeatures) {
  auto ctx = MakeCUDACtx(0);
  CheckShapUnusedFeatures(&ctx);
}

TEST(GPUPredictor, CompareCPUShap) {
  auto ctx = MakeCUDACtx(0);
  Context cpu_ctx;
  bst_feature_t constexpr kCols{10};
  bst_idx_t constexpr kRows{1000};
  std::size_t constexpr kIters{10};

  HostDeviceVector<float> predictions;
  HostDeviceVector<float> cpu_predictions;
  HostDeviceVector<float> interactions;
  HostDeviceVector<float> cpu_interactions;

  auto dmat = RandomDataGenerator(kRows, kCols, 0.0).Device(ctx.Device()).GenerateDMatrix();
  dmat->Info().labels.Reshape(kRows, 1);
  auto& h_labels = dmat->Info().labels.Data()->HostVector();
  for (size_t i = 0; i < kRows; ++i) {
    h_labels[i] = i % 2;
  }

  std::unique_ptr<Learner> learner{Learner::Create({dmat})};
  learner->Configure(Args{{"objective", "binary:logistic"},
                          {"max_depth", "12"},
                          {"min_split_loss", "0"},
                          {"min_child_weight", "0"},
                          {"reg_lambda", "0"},
                          {"reg_alpha", "0"},
                          {"subsample", "1"},
                          {"colsample_bytree", "1"},
                          {"device", ctx.DeviceName()}});
  learner->Configure();
  for (std::size_t i = 0; i < kIters; ++i) {
    learner->UpdateOneIter(i, dmat);
  }

  Json model{Object{}};
  learner->SaveModel(&model);

  std::unique_ptr<Learner> learner_gpu{Learner::Create({})};
  learner_gpu->LoadModel(model);
  learner_gpu->Configure({{"device", ctx.DeviceName()}});
  learner_gpu->Configure();

  std::unique_ptr<Learner> learner_cpu{Learner::Create({})};
  learner_cpu->LoadModel(model);
  learner_cpu->Configure({{"device", cpu_ctx.DeviceName()}});
  learner_cpu->Configure();

  learner_gpu->Predict(dmat, false, &predictions, 0, 0, false, false, true, false, false);
  learner_cpu->Predict(dmat, false, &cpu_predictions, 0, 0, false, false, true, false, false);
  learner_gpu->Predict(dmat, false, &interactions, 0, 0, false, false, false, false, true);
  learner_cpu->Predict(dmat, false, &cpu_interactions, 0, 0, false, false, false, false, true);
  ASSERT_TRUE(predictions.DeviceCanRead());
  ASSERT_TRUE(interactions.DeviceCanRead());
  EXPECT_THROW(
      learner_gpu->Predict(dmat, false, &predictions, 0, 0, false, false, true, true, false),
      dmlc::Error);
  EXPECT_THROW(
      learner_gpu->Predict(dmat, false, &interactions, 0, 0, false, false, false, true, true),
      dmlc::Error);
  auto& phis = predictions.HostVector();
  auto& cpu_phis = cpu_predictions.HostVector();
  for (auto i = 0ull; i < phis.size(); i++) {
    EXPECT_NEAR(cpu_phis[i], phis[i], 1e-4);
  }

  auto& inter = interactions.HostVector();
  auto& cpu_inter = cpu_interactions.HostVector();
  for (auto i = 0ull; i < inter.size(); i++) {
    EXPECT_NEAR(cpu_inter[i], inter[i], 1e-3);
  }
}

// Exercise independently advancing row subgroups under compute-sanitizer racecheck.
TEST(GPUPredictor, ShapPartialRowTiles) {
  auto ctx = MakeCUDACtx(0);
  bst_feature_t constexpr kCols{3};
  bst_idx_t constexpr kTrainRows{128};
  auto train = RandomDataGenerator(kTrainRows, kCols, 0.0).GenerateDMatrix();
  train->Info().labels.Reshape(kTrainRows, 1);
  auto& labels = train->Info().labels.Data()->HostVector();
  for (bst_idx_t i = 0; i < kTrainRows; ++i) {
    labels[i] = static_cast<float>(i % 7);
  }
  std::unique_ptr<Learner> cpu{Learner::Create({train})};
  cpu->Configure({{"tree_method", "hist"}, {"max_depth", "5"}, {"min_child_weight", "0"}});
  cpu->Configure();
  for (int i = 0; i < 3; ++i) {
    cpu->UpdateOneIter(i, train);
  }
  Json model{Object{}};
  cpu->SaveModel(&model);
  std::unique_ptr<Learner> gpu{Learner::Create({})};
  gpu->LoadModel(model);
  gpu->Configure({{"device", ctx.DeviceName()}});
  gpu->Configure();

  for (bst_idx_t rows : {1, 2, 3, 4, 5, 7, 8, 9}) {
    SCOPED_TRACE(rows);
    auto data = RandomDataGenerator(rows, kCols, 0.2).GenerateDMatrix();
    for (bool interactions : {false, true}) {
      SCOPED_TRACE(interactions);
      HostDeviceVector<float> expected, actual;
      cpu->Predict(data, false, &expected, 0, 0, false, false, !interactions, false, interactions);
      gpu->Predict(data, false, &actual, 0, 0, false, false, !interactions, false, interactions);
      ASSERT_TRUE(actual.DeviceCanRead());
      auto const& h_expected = expected.ConstHostVector();
      auto const& h_actual = actual.ConstHostVector();
      ASSERT_EQ(h_expected.size(), h_actual.size());
      for (std::size_t i = 0; i < h_expected.size(); ++i) {
        ASSERT_NEAR(h_expected[i], h_actual[i], 1e-4);
      }
    }
  }
}

TEST(GPUPredictor, ShapOutputCasesGPU) {
  auto ctx = MakeCUDACtx(0);
  auto cases = BuildShapTestCases(&ctx);
  for (auto const& [dmat, args] : cases) {
    CheckShapOutput(dmat.get(), args);
  }
}

TEST(GPUPredictor, DartShapOutputGPU) {
  auto ctx = MakeCUDACtx(0);
  CheckDartShapOutput(&ctx);
}

TEST(GPUPredictor, ShapHandlesDeepTree) {
  auto ctx = MakeCUDACtx(0);
  CheckShapHandlesDeepTree(&ctx);
}

TEST(GPUPredictor, ShapHandlesZeroCoverChild) {
  auto ctx = MakeCUDACtx(0);
  CheckShapHandlesZeroCover(&ctx, false);
}

TEST(GPUPredictor, ShapHandlesZeroCoverParent) {
  auto ctx = MakeCUDACtx(0);
  CheckShapHandlesZeroCover(&ctx, true);
}
}  // namespace xgboost
