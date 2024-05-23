/**
 * Copyright 2023, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/c_api.h>

#include <chrono>  // for ""s
#include <thread>  // for thread

#include "../../../src/collective/tracker.h"
#include "test_worker.h"   // for SocketTest
#include "xgboost/json.h"  // for Json

namespace xgboost::collective {
namespace {
class TrackerAPITest : public SocketTest {};
}  // namespace

TEST_F(TrackerAPITest, CAPI) {
  TrackerHandle handle;
  Json config{Object{}};
  config["dmlc_communicator"] = String{"rabit"};
  config["n_workers"] = 2;
  config["timeout"] = 1;
  auto config_str = Json::Dump(config);
  auto rc = XGTrackerCreate(config_str.c_str(), &handle);
  ASSERT_EQ(rc, 0);
  rc = XGTrackerRun(handle, nullptr);
  ASSERT_EQ(rc, 0);

  std::thread bg_wait{[&] {
    Json config{Object{}};
    auto config_str = Json::Dump(config);
    auto rc = XGTrackerWaitFor(handle, config_str.c_str());
    ASSERT_EQ(rc, 0);
  }};

  char const* cargs;
  rc = XGTrackerWorkerArgs(handle, &cargs);
  ASSERT_EQ(rc, 0);
  auto args = Json::Load(StringView{cargs});

  std::string host;
  ASSERT_TRUE(GetHostAddress(&host).OK());
  ASSERT_EQ(host, get<String const>(args["dmlc_tracker_uri"]));
  auto port = get<Integer const>(args["dmlc_tracker_port"]);
  ASSERT_NE(port, 0);

  std::vector<std::thread> workers;
  using namespace std::chrono_literals;  // NOLINT
  for (std::int32_t r = 0; r < 2; ++r) {
    workers.emplace_back([=] { WorkerForTest w{host, static_cast<std::int32_t>(port), 1s, 2, r}; });
  }
  for (auto& w : workers) {
    w.join();
  }

  rc = XGTrackerFree(handle);
  ASSERT_EQ(rc, 0);

  bg_wait.join();
}
}  // namespace xgboost::collective
