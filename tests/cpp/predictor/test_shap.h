/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#ifndef TESTS_CPP_PREDICTOR_TEST_SHAP_H_
#define TESTS_CPP_PREDICTOR_TEST_SHAP_H_

#include <xgboost/base.h>
#include <xgboost/context.h>
#include <xgboost/host_device_vector.h>

#include <memory>
#include <utility>
#include <vector>

namespace xgboost {
class DMatrix;
class Learner;
}  // namespace xgboost

namespace xgboost {
void CheckShapOutput(Context const* ctx, DMatrix* dmat, Args const& model_args);
void CheckShapAdditivity(size_t rows, size_t cols, HostDeviceVector<float> const& shap_values,
                         HostDeviceVector<float> const& margin_predt);

using ShapTestCase = std::pair<std::shared_ptr<DMatrix>, Args>;
std::vector<ShapTestCase> BuildShapTestCases(Context const* ctx);

}  // namespace xgboost

#endif  // TESTS_CPP_PREDICTOR_TEST_SHAP_H_
