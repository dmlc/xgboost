/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#ifndef XGBOOST_TEST_SHAP_H_
#define XGBOOST_TEST_SHAP_H_

#include <gtest/gtest.h>
#include <xgboost/context.h>

#include <tuple>

namespace xgboost {
class ShapExternalMemoryTest : public ::testing::TestWithParam<std::tuple<bool, bool>> {
 public:
  void Run(Context const* ctx, bool is_qdm, bool is_interaction);
};
}  // namespace xgboost

#endif  // XGBOOST_TEST_SHAP_H_
