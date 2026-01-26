/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#include "test_predictor.h"

#include <gtest/gtest.h>
#include <xgboost/data.h>                // for DMatrix
#include <xgboost/host_device_vector.h>  // for HostDeviceVector
#include <xgboost/json.h>                // for Json
#include <xgboost/predictor.h>           // for Predictor

#include <memory>  // for unique_ptr
#include <string>  // for to_string

#include "xgboost/linalg.h"  // for Tensor

namespace xgboost {
namespace {
void CheckBasicShap(DMatrix* dmat, Predictor* predictor, gbm::GBTreeModel const& model) {
  size_t const kRows = dmat->Info().num_row_;
  size_t const kCols = dmat->Info().num_col_;

  // Test predict contribution
  HostDeviceVector<float> out_contribution_hdv;
  auto& out_contribution = out_contribution_hdv.HostVector();
  predictor->PredictContribution(dmat, &out_contribution_hdv, model);
  ASSERT_EQ(out_contribution.size(), kRows * (kCols + 1));
  for (size_t i = 0; i < out_contribution.size(); ++i) {
    auto const& contri = out_contribution[i];
    // shift 1 for bias, as test tree is a decision dump, only global bias is
    // filled with LeafValue().
    if ((i + 1) % (kCols + 1) == 0) {
      ASSERT_EQ(out_contribution.back(), 1.5f);
    } else {
      ASSERT_EQ(contri, 0);
    }
  }

  // Test predict contribution (approximate method)
  predictor->PredictContribution(dmat, &out_contribution_hdv, model, 0, nullptr, true);
  for (size_t i = 0; i < out_contribution.size(); ++i) {
    auto const& contri = out_contribution[i];
    // shift 1 for bias, as test tree is a decision dump, only global bias is
    // filled with LeafValue().
    if ((i + 1) % (kCols + 1) == 0) {
      ASSERT_EQ(out_contribution.back(), 1.5f);
    } else {
      ASSERT_EQ(contri, 0);
    }
  }
}

void CheckTrainingPredictionShap(Learner* learner, std::shared_ptr<DMatrix> p_full,
                                 std::shared_ptr<DMatrix> p_hist) {
  // Contributions
  HostDeviceVector<float> from_full_contribs;
  learner->Predict(p_full, false, &from_full_contribs, 0, 0, false, false, true);
  HostDeviceVector<float> from_hist_contribs;
  learner->Predict(p_hist, false, &from_hist_contribs, 0, 0, false, false, true);
  for (size_t i = 0; i < from_full_contribs.ConstHostVector().size(); ++i) {
    EXPECT_NEAR(from_hist_contribs.ConstHostVector()[i], from_full_contribs.ConstHostVector()[i],
                kRtEps);
  }

  // Contributions (approximate method)
  HostDeviceVector<float> from_full_approx_contribs;
  learner->Predict(p_full, false, &from_full_approx_contribs, 0, 0, false, false, false, true);
  HostDeviceVector<float> from_hist_approx_contribs;
  learner->Predict(p_hist, false, &from_hist_approx_contribs, 0, 0, false, false, false, true);
  for (size_t i = 0; i < from_full_approx_contribs.ConstHostVector().size(); ++i) {
    EXPECT_NEAR(from_hist_approx_contribs.ConstHostVector()[i],
                from_full_approx_contribs.ConstHostVector()[i], kRtEps);
  }
}

std::unique_ptr<Learner> LearnerForShap(Context const* ctx, std::shared_ptr<DMatrix> dmat,
                                        size_t iters, size_t forest = 1) {
  std::unique_ptr<Learner> learner{Learner::Create({dmat})};
  learner->SetParams(Args{{"num_parallel_tree", std::to_string(forest)},
                          {"device", ctx->IsSycl() ? "cpu" : ctx->DeviceName()}});
  for (size_t i = 0; i < iters; ++i) {
    learner->UpdateOneIter(i, dmat);
  }

  return learner;
}

void CheckIterationRangeShap(std::shared_ptr<DMatrix> dmat, Learner* learner, Learner* sliced,
                             bst_layer_t lend) {
  HostDeviceVector<float> out_predt_sliced;
  HostDeviceVector<float> out_predt_ranged;

  // SHAP
  {
    sliced->Predict(dmat, false, &out_predt_sliced, 0, 0, false, false, true, false, false);
    learner->Predict(dmat, false, &out_predt_ranged, 0, lend, false, false, true, false, false);

    auto const& h_sliced = out_predt_sliced.HostVector();
    auto const& h_range = out_predt_ranged.HostVector();
    ASSERT_EQ(h_sliced.size(), h_range.size());
    ASSERT_EQ(h_sliced, h_range);
  }

  // SHAP interaction
  {
    sliced->Predict(dmat, false, &out_predt_sliced, 0, 0, false, false, false, false, true);
    learner->Predict(dmat, false, &out_predt_ranged, 0, lend, false, false, false, false, true);
    auto const& h_sliced = out_predt_sliced.HostVector();
    auto const& h_range = out_predt_ranged.HostVector();
    ASSERT_EQ(h_sliced.size(), h_range.size());
    ASSERT_EQ(h_sliced, h_range);
  }
}
}  // anonymous namespace

TEST(Predictor, ShapBasic) {
  Context ctx;
  size_t constexpr kRows = 5;
  size_t constexpr kCols = 5;
  auto dmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();

  auto predictor = std::unique_ptr<Predictor>(CreatePredictorForTest(&ctx));
  LearnerModelParam mparam{MakeMP(kCols, .0, 1, ctx.Device())};
  std::unique_ptr<gbm::GBTreeModel> p_model = CreateTestModel(&mparam, &ctx);
  auto const& model = *p_model;

  CheckBasicShap(dmat.get(), predictor.get(), model);
}

TEST(Predictor, ShapTrainingPrediction) {
  Context ctx;
  size_t constexpr kRows = 1000;
  size_t constexpr kCols = 16;
  size_t constexpr kClasses = 3;
  size_t constexpr kBins = 64;
  size_t constexpr kIters = 3;

  auto p_full = RandomDataGenerator{kRows, kCols, 0.0}.GenerateDMatrix(true);
  auto p_hist = RandomDataGenerator{kRows, kCols, 0.0}.Bins(kBins).GenerateQuantileDMatrix(false);

  p_hist->Info().labels.Reshape(kRows, 1);
  auto& h_label = p_hist->Info().labels.Data()->HostVector();
  for (size_t i = 0; i < kRows; ++i) {
    h_label[i] = i % kClasses;
  }

  std::unique_ptr<Learner> learner;
  learner.reset(Learner::Create({}));
  learner->SetParams(Args{{"objective", "multi:softprob"},
                          {"num_feature", std::to_string(kCols)},
                          {"num_class", std::to_string(kClasses)},
                          {"max_bin", std::to_string(kBins)},
                          {"device", ctx.DeviceName()}});
  learner->Configure();

  for (size_t i = 0; i < kIters; ++i) {
    learner->UpdateOneIter(i, p_hist);
  }

  Json model{Object{}};
  learner->SaveModel(&model);

  learner.reset(Learner::Create({}));
  learner->LoadModel(model);
  learner->SetParam("device", ctx.DeviceName());
  learner->Configure();

  CheckTrainingPredictionShap(learner.get(), p_full, p_hist);
}

TEST(Predictor, ShapIterationRange) {
  Context ctx;
  size_t constexpr kRows = 1000;
  size_t constexpr kCols = 20;
  size_t constexpr kClasses = 4;
  size_t constexpr kForest = 3;
  size_t constexpr kIters = 10;

  auto dmat = RandomDataGenerator(kRows, kCols, 0)
                  .Device(ctx.Device())
                  .Classes(kClasses)
                  .GenerateDMatrix(true);
  auto learner = LearnerForShap(&ctx, dmat, kIters, kForest);

  bool bound = false;
  bst_layer_t lend{3};
  std::unique_ptr<Learner> sliced{learner->Slice(0, lend, 1, &bound)};
  ASSERT_FALSE(bound);

  CheckIterationRangeShap(dmat, learner.get(), sliced.get(), lend);
}

void ShapExternalMemoryTest::Run(Context const* ctx, bool is_qdm, bool is_interaction) {
  bst_idx_t n_samples{2048};
  bst_feature_t n_features{16};
  bst_target_t n_classes{3};
  bst_bin_t max_bin{64};
  auto create_pfmat = [&](RandomDataGenerator& rng) {
    if (is_qdm) {
      return rng.Bins(max_bin).GenerateExtMemQuantileDMatrix("temp", true);
    }
    return rng.GenerateSparsePageDMatrix("temp", true);
  };
  auto p_fmat = create_pfmat(RandomDataGenerator(n_samples, n_features, 0)
                                 .Batches(1)
                                 .Device(ctx->Device())
                                 .Classes(n_classes));
  std::unique_ptr<Learner> learner{Learner::Create({p_fmat})};
  learner->SetParam("device", ctx->DeviceName());
  learner->SetParam("base_score", "[0.5, 0.5, 0.5]");
  learner->SetParam("num_parallel_tree", "3");
  learner->SetParam("max_bin", std::to_string(max_bin));
  for (std::int32_t i = 0; i < 4; ++i) {
    learner->UpdateOneIter(i, p_fmat);
  }
  Json model{Object{}};
  learner->SaveModel(&model);
  auto j_booster = model["learner"]["gradient_booster"]["model"];

  auto base_score = linalg::Tensor<float, 1>{{0.0, 0.0, 0.0}, {3}, ctx->Device()};
  LearnerModelParam model_param(n_features, std::move(base_score), n_classes, 1,
                                MultiStrategy::kOneOutputPerTree);
  gbm::GBTreeModel gbtree{&model_param, ctx};
  gbtree.LoadModel(j_booster);

  std::unique_ptr<Predictor> predictor{
      Predictor::Create(ctx->IsCPU() ? "cpu_predictor" : "gpu_predictor", ctx)};
  predictor->Configure({});
  HostDeviceVector<float> contrib;
  if (is_interaction) {
    predictor->PredictInteractionContributions(p_fmat.get(), &contrib, gbtree);
  } else {
    predictor->PredictContribution(p_fmat.get(), &contrib, gbtree);
  }

  auto p_fmat_ext = create_pfmat(RandomDataGenerator(n_samples, n_features, 0)
                                     .Batches(4)
                                     .Device(ctx->Device())
                                     .Classes(n_classes));

  HostDeviceVector<float> contrib_ext;
  if (is_interaction) {
    predictor->PredictInteractionContributions(p_fmat_ext.get(), &contrib_ext, gbtree);
  } else {
    predictor->PredictContribution(p_fmat_ext.get(), &contrib_ext, gbtree);
  }

  ASSERT_EQ(contrib_ext.Size(), contrib.Size());

  auto h_contrib = contrib.ConstHostSpan();
  auto h_contrib_ext = contrib_ext.ConstHostSpan();
  for (std::size_t i = 0; i < h_contrib.size(); ++i) {
    ASSERT_EQ(h_contrib[i], h_contrib_ext[i]);
  }
}

INSTANTIATE_TEST_SUITE_P(Predictor, ShapExternalMemoryTest,
                         ::testing::Combine(::testing::Bool(), ::testing::Bool()));
}  // namespace xgboost
