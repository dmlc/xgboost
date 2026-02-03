/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#include "test_shap.h"

#include <gtest/gtest.h>
#include <xgboost/base.h>
#include <xgboost/data.h>                // for DMatrix
#include <xgboost/host_device_vector.h>  // for HostDeviceVector
#include <xgboost/json.h>                // for Json
#include <xgboost/learner.h>             // for Learner
#include <xgboost/linalg.h>              // for Vector

#include <algorithm>
#include <memory>  // for unique_ptr
#include <sstream>
#include <string>  // for to_string

#include "../../../src/common/param_array.h"
#include "../../../src/gbm/gbtree_model.h"
#include "../../../src/interpretability/shap.h"
#include "../helpers.h"

namespace xgboost {
namespace {
void SetLabels(DMatrix* dmat, bst_target_t n_classes) {
  size_t const rows = dmat->Info().num_row_;
  dmat->Info().labels.Reshape(rows, 1);
  auto& h_labels = dmat->Info().labels.Data()->HostVector();
  if (n_classes > 1) {
    for (size_t i = 0; i < rows; ++i) {
      h_labels[i] = static_cast<float>(i % n_classes);
    }
  } else {
    for (size_t i = 0; i < rows; ++i) {
      h_labels[i] = static_cast<float>(i % 2);
    }
  }
}

Args BaseParams(Context const* ctx, std::string objective, std::string max_depth) {
  return Args{{"objective", std::move(objective)},
              {"max_depth", std::move(max_depth)},
              {"min_split_loss", "0"},
              {"min_child_weight", "0"},
              {"reg_lambda", "0"},
              {"reg_alpha", "0"},
              {"subsample", "1"},
              {"colsample_bytree", "1"},
              {"device", ctx->IsSycl() ? "cpu" : ctx->DeviceName()}};
}

gbm::GBTreeModel LoadGBTreeModel(Learner* learner, Context const* ctx,
                                 LearnerModelParam* out_param) {
  Json model{Object{}};
  learner->SaveModel(&model);

  CHECK(IsA<Object>(model)) << model;
  auto const& model_obj = get<Object const>(model);
  auto learner_it = model_obj.find("learner");
  CHECK(learner_it != model_obj.cend()) << model;
  CHECK(IsA<Object>(learner_it->second)) << model;
  auto const& learner_obj = get<Object const>(learner_it->second);

  auto const& lmp = get<Object const>(learner_obj.at("learner_model_param"));
  auto const& num_feature = get<String const>(lmp.at("num_feature"));
  auto const& num_class = get<String const>(lmp.at("num_class"));
  auto const& num_target = get<String const>(lmp.at("num_target"));
  auto const& base_score_str = get<String const>(lmp.at("base_score"));

  common::ParamArray<float> base_score_arr{"base_score"};
  std::stringstream ss;
  ss << base_score_str;
  ss >> base_score_arr;

  std::size_t shape[1]{base_score_arr.size()};
  linalg::Vector<float> base_score_vec{shape, ctx->Device()};
  auto& h_base = base_score_vec.Data()->HostVector();
  h_base.assign(base_score_arr.cbegin(), base_score_arr.cend());

  auto n_features = static_cast<bst_feature_t>(std::stol(num_feature));
  auto n_classes = static_cast<bst_target_t>(std::stol(num_class));
  auto n_targets = static_cast<bst_target_t>(std::stol(num_target));
  auto n_groups = static_cast<uint32_t>(std::max(n_classes, n_targets));
  *out_param = LearnerModelParam{n_features, std::move(base_score_vec), n_groups, n_targets,
                                 MultiStrategy::kOneOutputPerTree};

  gbm::GBTreeModel gbtree{out_param, ctx};
  auto gbm_it = learner_obj.find("gradient_booster");
  CHECK(gbm_it != learner_obj.cend()) << model;
  CHECK(IsA<Object>(gbm_it->second)) << model;
  gbtree.LoadModel(gbm_it->second);
  return gbtree;
}
}  // namespace

std::vector<ShapTestCase> BuildShapTestCases(Context const* ctx) {
  std::vector<ShapTestCase> cases;
  auto device = ctx->Device();

  {
    // small dense, shallow tree
    auto dmat = RandomDataGenerator(32, 6, 0.0).Device(device).GenerateDMatrix();
    SetLabels(dmat.get(), 1);
    cases.emplace_back(dmat, BaseParams(ctx, "reg:squarederror", "2"));
  }

  {
    // medium dense training DMatrix, moderate depth
    auto dmat = RandomDataGenerator(512, 10, 0.0).Device(device).GenerateDMatrix(true);
    SetLabels(dmat.get(), 1);
    cases.emplace_back(dmat, BaseParams(ctx, "reg:squarederror", "6"));
  }

  {
    // quantile DMatrix with explicit bins, deeper tree
    auto dmat =
        RandomDataGenerator(2048, 12, 0.0).Bins(64).Device(device).GenerateQuantileDMatrix(false);
    SetLabels(dmat.get(), 1);
    auto args = BaseParams(ctx, "reg:squarederror", "8");
    args.emplace_back("max_bin", "64");
    cases.emplace_back(dmat, std::move(args));
  }

  {
    // external memory quantile DMatrix, moderate depth
    bst_bin_t max_bin{64};
    auto dmat = RandomDataGenerator(4096, 10, 0.0)
                    .Batches(2)
                    .Bins(max_bin)
                    .Device(device)
                    .GenerateExtMemQuantileDMatrix("shap_extmem", true);
    SetLabels(dmat.get(), 1);
    auto args = BaseParams(ctx, "reg:squarederror", "6");
    args.emplace_back("max_bin", std::to_string(max_bin));
    cases.emplace_back(dmat, std::move(args));
  }

  {
    // external memory sparse page DMatrix, moderate depth
    auto dmat = RandomDataGenerator(4096, 10, 0.0)
                    .Batches(2)
                    .Device(device)
                    .GenerateSparsePageDMatrix("shap_extmem", true);
    SetLabels(dmat.get(), 1);
    cases.emplace_back(dmat, BaseParams(ctx, "reg:squarederror", "6"));
  }

  {
    // multi-class dense training DMatrix, medium depth
    bst_target_t n_classes{3};
    auto dmat =
        RandomDataGenerator(256, 8, 0.0).Classes(n_classes).Device(device).GenerateDMatrix(true);
    SetLabels(dmat.get(), n_classes);
    auto args = BaseParams(ctx, "multi:softprob", "4");
    args.emplace_back("num_class", std::to_string(n_classes));
    cases.emplace_back(dmat, std::move(args));
  }

  {
    // large dense, deeper tree and classification objective
    auto dmat = RandomDataGenerator(10000, 12, 0.0).Device(device).GenerateDMatrix();
    SetLabels(dmat.get(), 1);
    cases.emplace_back(dmat, BaseParams(ctx, "binary:logistic", "10"));
  }

  return cases;
}

void CheckShapOutput(DMatrix* dmat, Args const& model_args) {
  size_t const kRows = dmat->Info().num_row_;
  size_t const kCols = dmat->Info().num_col_;

  std::shared_ptr<DMatrix> p_dmat{dmat, [](DMatrix*) {}};
  std::unique_ptr<Learner> learner{Learner::Create({p_dmat})};
  learner->SetParams(model_args);
  learner->Configure();
  for (size_t i = 0; i < 5; ++i) {
    learner->UpdateOneIter(i, p_dmat);
  }

  HostDeviceVector<float> margin_predt;
  learner->Predict(p_dmat, true, &margin_predt, 0, 0, false, false, false, false, false);
  size_t const n_outputs = margin_predt.HostVector().size() / kRows;

  LearnerModelParam mparam;
  auto gbtree = LoadGBTreeModel(learner.get(), dmat->Ctx(), &mparam);

  HostDeviceVector<float> shap_values;
  interpretability::ShapValues(dmat->Ctx(), p_dmat.get(), &shap_values, gbtree, 0, nullptr, 0, 0);
  ASSERT_EQ(shap_values.HostVector().size(), kRows * (kCols + 1) * n_outputs);
  CheckShapAdditivity(kRows, kCols, shap_values, margin_predt);

  HostDeviceVector<float> shap_interactions;
  interpretability::ShapInteractionValues(dmat->Ctx(), p_dmat.get(), &shap_interactions, gbtree, 0,
                                          nullptr, false);
  ASSERT_EQ(shap_interactions.HostVector().size(), kRows * (kCols + 1) * (kCols + 1) * n_outputs);
  CheckShapAdditivity(kRows, kCols, shap_interactions, margin_predt);
}

void CheckShapAdditivity(size_t rows, size_t cols, HostDeviceVector<float> const& shap_values,
                         HostDeviceVector<float> const& margin_predt) {
  auto const& h_shap = shap_values.ConstHostVector();
  auto const& h_margin = margin_predt.ConstHostVector();

  ASSERT_EQ(h_margin.size() % rows, 0);
  size_t const n_outputs = h_margin.size() / rows;
  size_t const kShapSize = rows * (cols + 1) * n_outputs;
  size_t const kInteractionSize = rows * (cols + 1) * (cols + 1) * n_outputs;
  bool const is_interaction = h_shap.size() == kInteractionSize;
  ASSERT_TRUE(h_shap.size() == kShapSize || is_interaction);

  for (size_t row = 0; row < rows; ++row) {
    for (size_t out = 0; out < n_outputs; ++out) {
      float sum = 0.0f;
      if (is_interaction) {
        size_t const base = (row * n_outputs + out) * (cols + 1) * (cols + 1);
        for (size_t idx = 0; idx < (cols + 1) * (cols + 1); ++idx) {
          sum += h_shap[base + idx];
        }
      } else {
        size_t const base = (row * n_outputs + out) * (cols + 1);
        for (size_t c = 0; c < cols + 1; ++c) {
          sum += h_shap[base + c];
        }
      }
      EXPECT_NEAR(sum, h_margin[row * n_outputs + out], 1e-5f);
    }
  }
}

TEST(Predictor, ShapOutputCasesCPU) {
  Context ctx;
  auto cases = BuildShapTestCases(&ctx);
  for (auto const& [dmat, args] : cases) {
    CheckShapOutput(dmat.get(), args);
  }
}

TEST(Predictor, ApproxContribsBasic) {
  Context ctx;
  size_t constexpr kRows = 64;
  size_t constexpr kCols = 6;

  auto dmat = RandomDataGenerator(kRows, kCols, 0.0).Device(ctx.Device()).GenerateDMatrix();
  SetLabels(dmat.get(), 1);

  auto args = BaseParams(&ctx, "reg:squarederror", "3");
  args.emplace_back("tree_method", "approx");

  std::unique_ptr<Learner> learner{Learner::Create({dmat})};
  learner->SetParams(args);
  learner->Configure();
  for (size_t i = 0; i < 3; ++i) {
    learner->UpdateOneIter(i, dmat);
  }

  HostDeviceVector<float> margin_predt;
  learner->Predict(dmat, true, &margin_predt, 0, 0, false, false, false, false, false);

  LearnerModelParam mparam;
  auto gbtree = LoadGBTreeModel(learner.get(), dmat->Ctx(), &mparam);

  HostDeviceVector<float> approx_contribs;
  interpretability::ApproxFeatureImportance(dmat->Ctx(), dmat.get(), &approx_contribs, gbtree, 0,
                                            nullptr);

  auto const& h_margin = margin_predt.ConstHostVector();
  auto const& h_contribs = approx_contribs.ConstHostVector();
  ASSERT_EQ(h_margin.size(), kRows);
  ASSERT_EQ(h_contribs.size(), kRows * (kCols + 1));

  for (size_t row = 0; row < kRows; ++row) {
    float sum = 0.0f;
    size_t base = row * (kCols + 1);
    for (size_t c = 0; c < kCols + 1; ++c) {
      sum += h_contribs[base + c];
    }
    EXPECT_NEAR(sum, h_margin[row], 1e-2f);
  }
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
  std::unique_ptr<Learner> learner{Learner::Create({dmat})};
  learner->SetParams(Args{{"num_parallel_tree", std::to_string(kForest)},
                          {"device", ctx.IsSycl() ? "cpu" : ctx.DeviceName()}});
  for (size_t i = 0; i < kIters; ++i) {
    learner->UpdateOneIter(i, dmat);
  }

  bool bound = false;
  bst_layer_t lend{3};
  std::unique_ptr<Learner> sliced{learner->Slice(0, lend, 1, &bound)};
  ASSERT_FALSE(bound);

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

}  // namespace xgboost
