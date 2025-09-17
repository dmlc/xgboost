/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <memory>  // for make_unique
#include <string>  // for string

#include "../../../../src/collective/tracker.h"  // for GetHostAddress
#include "federated_tracker.h"
#include "xgboost/json.h"  // for Json

namespace xgboost::collective {
TEST(FederatedTrackerTest, Basic) {
  Json config{Object()};
  config["federated_secure"] = Boolean{false};
  config["n_workers"] = Integer{3};

  auto tracker = std::make_unique<FederatedTracker>(config);
  ASSERT_FALSE(tracker->Ready());
  auto fut = tracker->Run();
  auto args = tracker->WorkerArgs();
  ASSERT_TRUE(tracker->Ready());

  ASSERT_GE(tracker->Port(), 1);
  std::string host;
  auto rc = GetHostAddress(&host);
  ASSERT_EQ(get<String const>(args["dmlc_tracker_uri"]), host);

  rc = tracker->Shutdown();
  SafeColl(rc);
  SafeColl(fut.get());
  ASSERT_FALSE(tracker->Ready());
}
}  // namespace xgboost::collective
