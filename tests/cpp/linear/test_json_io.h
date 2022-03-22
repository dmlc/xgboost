/*!
 * Copyright 2020 XGBoost contributors
 */
#ifndef XGBOOST_TEST_JSON_IO_H_
#define XGBOOST_TEST_JSON_IO_H_

#include <xgboost/linear_updater.h>
#include <xgboost/json.h>
#include <string>
#include "../helpers.h"
#include "../../../src/gbm/gblinear_model.h"

namespace xgboost {
inline void TestUpdaterJsonIO(std::string updater_str) {
  auto runtime = xgboost::CreateEmptyGenericParam(GPUIDX);
  Json config_0 {Object() };

  {
    auto updater = std::unique_ptr<xgboost::LinearUpdater>(
        xgboost::LinearUpdater::Create(updater_str, &runtime));
    updater->Configure({{"eta", std::to_string(3.14)}});
    updater->SaveConfig(&config_0);
  }

  {
    auto updater = std::unique_ptr<xgboost::LinearUpdater>(
        xgboost::LinearUpdater::Create(updater_str, &runtime));
    updater->LoadConfig(config_0);
    Json config_1 { Object() };
    updater->SaveConfig(&config_1);

    ASSERT_EQ(config_0, config_1);
    auto eta = atof(get<String const>(config_1["linear_train_param"]["eta"]).c_str());
    ASSERT_NEAR(eta, 3.14, kRtEps);
  }

}

}  // namespace xgboost

#endif  // XGBOOST_TEST_JSON_IO_H_
