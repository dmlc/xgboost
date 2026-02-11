/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#pragma once

#include <xgboost/context.h>  // for Context

#include <string>  // for string
#include <vector>  // for vector

namespace xgboost {

void TestLinearRegressionGPair(const Context* ctx);

void TestSquaredLog(const Context* ctx);

void TestLogisticRegressionGPair(const Context* ctx);

void TestLogisticRegressionBasic(const Context* ctx);

void TestsLogisticRawGPair(const Context* ctx);

void TestPoissonRegressionGPair(const Context* ctx);

void TestPoissonRegressionBasic(const Context* ctx);

void TestGammaRegressionGPair(const Context* ctx);

void TestGammaRegressionBasic(const Context* ctx);

void TestTweedieRegressionGPair(const Context* ctx);

void TestTweedieRegressionBasic(const Context* ctx);

void TestCoxRegressionGPair(const Context* ctx);

void TestAbsoluteError(const Context* ctx);

void TestAbsoluteErrorLeaf(const Context* ctx);

void TestVectorLeafObj(Context const* ctx, std::string name, Args const& args, bst_idx_t n_samples,
                       bst_idx_t n_target_labels, std::vector<float> const& sol_left,
                       std::vector<float> const& sol_right);

void TestPseudoHuber(const Context* ctx);

void TestExpectileRegressionGPair(const Context* ctx);

void TestExpectileRegressionMultiAlpha(const Context* ctx);

void TestExpectileRegressionInitEstimation(const Context* ctx);

}  // namespace xgboost
