/**
 * Copyright 2024, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include "../../../../plugin/federated/federated_plugin.h"
#include "xgboost/json.h"

namespace xgboost::collective {
TEST(FederatedPluginMock, Basic) {
  Json config{Object{}};
  config["federated_plugin"] = Object{};
  config["federated_plugin"]["name"] = String{"mock"};
  std::unique_ptr<FederatedPluginBase> plugin{CreateFederatedPlugin(config)};

  bst_idx_t n_bins{16};
  std::vector<double> hist(n_bins, 1.0);
  auto enc_hist = plugin->BuildEncryptedHistHori(hist);
  auto plain_hist = plugin->SyncEncryptedHistHori(enc_hist);
  ASSERT_EQ(hist.size(), plain_hist.size());
  for (std::size_t i = 0; i < hist.size(); ++i) {
    ASSERT_EQ(plain_hist[i], hist[i]);
  }
}
}  // namespace xgboost::collective
