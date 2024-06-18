/**
 * Copyright 2023, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <memory>  // for make_unique
#include <string>  // for string

#include "../../../../src/collective/tracker.h"  // for GetHostAddress
#include "federated_tracker.h"
#include "test_worker.h"
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
  ASSERT_EQ(get<String const>(args["DMLC_TRACKER_URI"]), host);

  rc = tracker->Shutdown();
  ASSERT_TRUE(rc.OK());
  ASSERT_TRUE(fut.get().OK());
  ASSERT_FALSE(tracker->Ready());
}
}  // namespace xgboost::collective
