/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>  // for size_t
#include <cstdint>  // for int32_t
#include <memory>   // for make_unique, shared_ptr, unique_ptr
#include <numeric>  // for accumulate
#include <string>   // for string
#include <vector>   // for vector

#include "../../../src/cross_validate/cross_validate.h"  // for FoldEvaluator
#include "../../../src/cross_validate/kfolds.h"          // for KFold
#include "../helpers.h"  // for GMockThrow, MakeCUDACtx, RandomDataGenerator
#include "xgboost/json.h"

// Only what `tests/python-gpu/test_cross_validate.py` cannot reach; it owns the values, the
// names, and every error a driver can provoke through the C API.
namespace xgboost::cv {
namespace {
constexpr float kPredt = 0.25f;
constexpr float kLabel = 0.5f;
constexpr double kResidual = 0.25;

// A run sized but never grown, over a constant prediction and a constant label, so that any
// evaluation reaching a metric must report `kResidual` for every fold and both splits.
struct EvalRun {
  Context ctx{MakeCUDACtx(0)};
  std::shared_ptr<DMatrix> p_fmat;
  std::unique_ptr<FoldModels> models;
  FoldInfoBatches finfo;
  FoldPredictions predts;

  // `batch_sizes` sets the per-batch fold windows, which external memory does not let Python
  // line up this way.
  EvalRun(std::vector<bst_idx_t> const& batch_sizes, std::size_t k_folds) {
    auto n_rows = std::accumulate(batch_sizes.cbegin(), batch_sizes.cend(), bst_idx_t{0});
    // On device: `FoldModels` copies the matrix's context and the evaluator reads it back.
    this->p_fmat = RandomDataGenerator{n_rows, 4, 0.0f}
                       .Device(this->ctx.Device())
                       .Bins(16)
                       .GenerateQuantileDMatrix(false);
    this->p_fmat->Info().labels.Reshape(n_rows, 1);
    this->p_fmat->Info().labels.Data()->Fill(kLabel);
    this->models = std::make_unique<FoldModels>(k_folds, this->p_fmat, false);

    bst_idx_t begin = 0;
    for (auto size : batch_sizes) {
      this->finfo.batches.emplace_back();
      for (std::size_t k = 0; k < k_folds; ++k) {
        KFold(&this->ctx, k_folds, begin, begin + size, static_cast<std::int32_t>(k),
              &this->finfo.batches.back());
      }
      begin += size;
    }

    this->predts.layout = this->models->Layout();
    this->predts.output_length = this->models->OutputLength(0);
    this->predts.train.resize(this->models->NumUnits());
    for (auto& train : this->predts.train) {
      this->Size(n_rows, &train);
    }
    this->Size(n_rows, &this->predts.valid);
  }

  FoldEvalResult const& Eval(FoldEvaluator* p_eval) const {
    return p_eval->Eval(*this->models, this->p_fmat->Info(), this->finfo, this->predts, 0);
  }

 private:
  void Size(bst_idx_t n_rows, gbm::PredictionCacheEntry* out) {
    out->predictions.SetDevice(this->ctx.Device());
    out->predictions.Resize(n_rows * this->predts.output_length);
    out->predictions.Fill(kPredt);
    // Posing as the output of round 0, which is what `Eval` is told it is evaluating.
    out->version = 1;
  }
};

[[nodiscard]] Json MakeConfig(std::string const& metric) {
  Json config{Object{}};
  config["eval_metric"] = Array{std::vector<Json>{Json{String{metric}}}};
  config["eval_train"] = Boolean{true};
  return config;
}
}  // namespace

// Two batches of three rows with five folds starves folds 3 and 4 globally: `cv::KFold` gives
// every fold at or above the batch row count an empty window.
TEST(FoldEvaluator, StarvedFold) {
  EvalRun run{{3, 3}, 5};
  ASSERT_EQ(run.finfo.ValidFoldSize(4), 0);
  FoldEvaluator evaluator{*run.models, MakeConfig("rmse")};
  ASSERT_THAT([&] { run.Eval(&evaluator); }, GMockThrow("holds out no row"));
}

// A batch shorter than `k_folds` leaves a fold an empty window there, which the gather must
// skip rather than read as "every row". Python cannot see a per-batch window to pin this.
TEST(FoldEvaluator, EmptyWindow) {
  EvalRun run{{9, 2}, 3};
  ASSERT_TRUE(run.finfo.batches.back().ValidationFold(2).empty());
  FoldEvaluator evaluator{*run.models, MakeConfig("rmse")};
  auto const& result = run.Eval(&evaluator);
  ASSERT_EQ(result.values.size(), 2 * run.finfo.KFolds());
  for (auto value : result.values) {
    EXPECT_NEAR(value, kResidual, 1e-6);
  }
}

// Python has no `params` argument left to send a metric parameter through.
TEST(FoldEvaluator, StrayParameter) {
  EvalRun run{{16}, 3};
  auto config = MakeConfig("rmse");
  config["huber_slope"] = Number{2.0};
  auto build = [&] {
    FoldEvaluator evaluator{*run.models, config};
  };
  ASSERT_THAT(build, GMockThrow("Unknown CV evaluator parameter"));
}
}  // namespace xgboost::cv
