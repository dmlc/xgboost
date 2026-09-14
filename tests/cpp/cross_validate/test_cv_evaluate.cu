/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <cstddef>  // for size_t
#include <cstdint>  // for int32_t
#include <memory>   // for make_unique, shared_ptr, unique_ptr
#include <string>   // for string
#include <vector>   // for vector

#include "../../../src/common/device_helpers.cuh"
#include "../../../src/cross_validate/cross_validate.h"  // for FoldEvaluator
#include "../helpers.h"  // for GMockThrow, MakeCUDACtx, RandomDataGenerator
#include "xgboost/json.h"

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
  FoldPredictions predts;

  EvalRun(bst_idx_t n_rows, std::size_t k_folds) {
    // On device: `FoldModels` copies the matrix's context and the evaluator reads it back.
    this->p_fmat = RandomDataGenerator{n_rows, 4, 0.0f}
                       .Device(this->ctx.Device())
                       .Bins(16)
                       .GenerateQuantileDMatrix(false);
    this->p_fmat->Info().labels.Reshape(n_rows, 1);
    this->p_fmat->Info().labels.Data()->Fill(kLabel);
    this->models = std::make_unique<FoldModels>(k_folds, this->p_fmat, false);

    dh::DeviceUVector<std::int64_t> ids(n_rows);
    auto first = thrust::make_counting_iterator(std::int64_t{0});
    thrust::transform(ctx.CUDACtx()->CTP(), first, first + n_rows,
                      thrust::make_constant_iterator(k_folds), ids.data(),
                      thrust::modulus<std::int64_t>{});
    this->predts.assignment = std::make_shared<FoldAssignment>(&ctx, k_folds, dh::ToSpan(ids));
    this->predts.layout = this->models->Layout();
    this->predts.output_length = this->models->OutputLength(0);
    this->predts.train.resize(this->models->NumUnits());
    for (auto& train : this->predts.train) {
      this->Size(n_rows, &train);
    }
    this->Size(n_rows, &this->predts.valid);
  }

  FoldEvalResult const& Eval(FoldEvaluator* p_eval) const {
    return p_eval->Eval(*this->models, this->p_fmat->Info(), this->predts, 0);
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

// All five folds have validation rows, including four single-row folds.
TEST(FoldEvaluator, TinyFolds) {
  EvalRun run{6, 5};
  FoldEvaluator evaluator{*run.models, MakeConfig("rmse")};
  auto const& result = run.Eval(&evaluator);
  for (auto value : result.values) {
    EXPECT_NEAR(value, kResidual, 1e-6);
  }
}

// Reusing scratch for different fold sizes must not retain rows from a previous fold.
TEST(FoldEvaluator, UnevenFolds) {
  EvalRun run{11, 3};
  ASSERT_EQ(run.predts.Assignment().ValidFoldSize(2), 3);
  FoldEvaluator evaluator{*run.models, MakeConfig("rmse")};
  auto const& result = run.Eval(&evaluator);
  ASSERT_EQ(result.values.size(), 2 * run.predts.Assignment().KFolds());
  for (auto value : result.values) {
    EXPECT_NEAR(value, kResidual, 1e-6);
  }
}

TEST(FoldEvaluator, StrayParameter) {
  EvalRun run{16, 3};
  auto config = MakeConfig("rmse");
  config["huber_slope"] = Number{2.0};
  auto build = [&] {
    FoldEvaluator evaluator{*run.models, config};
  };
  ASSERT_THAT(build, GMockThrow("Unknown CV evaluator parameter"));
}
}  // namespace xgboost::cv
