/**
 * Copyright 2022-2023, XGBoost contributors
 */
#pragma once

#include <gtest/gtest.h>

#include <chrono>  // for ms, seconds
#include <memory>  // for shared_ptr
#include <thread>  // for thread

#include "../../../../plugin/federated/federated_tracker.h"
#include "../../../../src/collective/comm_group.h"
#include "federated_comm.h"  // for FederatedComm
#include "xgboost/json.h"    // for Json

namespace xgboost::collective {
template <typename WorkerFn>
void TestFederated(std::int32_t n_workers, WorkerFn&& fn) {
  Json config{Object()};
  config["federated_secure"] = Boolean{false};
  config["n_workers"] = Integer{n_workers};
  FederatedTracker tracker{config};
  auto fut = tracker.Run();

  std::vector<std::thread> workers;
  using namespace std::chrono_literals;
  auto rc = tracker.WaitUntilReady();
  ASSERT_TRUE(rc.OK()) << rc.Report();
  std::int32_t port = tracker.Port();

  for (std::int32_t i = 0; i < n_workers; ++i) {
    workers.emplace_back([=] {
      Json config{Object{}};
      config["federated_world_size"] = n_workers;
      config["federated_rank"] = i;
      config["federated_server_address"] = "0.0.0.0:" + std::to_string(port);
      auto comm = std::make_shared<FederatedComm>(
          DefaultRetry(), std::chrono::seconds{DefaultTimeoutSec()}, std::to_string(i), config);

      fn(comm, i);
    });
  }

  for (auto& t : workers) {
    t.join();
  }

  rc = tracker.Shutdown();
  ASSERT_TRUE(rc.OK()) << rc.Report();
  ASSERT_TRUE(fut.get().OK());
}

template <typename WorkerFn>
void TestFederatedGroup(std::int32_t n_workers, WorkerFn&& fn) {
  Json config{Object()};
  config["federated_secure"] = Boolean{false};
  config["n_workers"] = Integer{n_workers};
  FederatedTracker tracker{config};
  auto fut = tracker.Run();

  std::vector<std::thread> workers;
  auto rc = tracker.WaitUntilReady();
  ASSERT_TRUE(rc.OK()) << rc.Report();
  std::int32_t port = tracker.Port();

  for (std::int32_t i = 0; i < n_workers; ++i) {
    workers.emplace_back([=] {
      Json config{Object{}};
      config["dmlc_communicator"] = std::string{"federated"};
      config["dmlc_task_id"] = std::to_string(i);
      config["dmlc_retry"] = 2;
      config["federated_world_size"] = n_workers;
      config["federated_rank"] = i;
      config["federated_server_address"] = "0.0.0.0:" + std::to_string(port);
      std::shared_ptr<CommGroup> comm_group{CommGroup::Create(config)};
      fn(comm_group, i);
    });
  }

  for (auto& t : workers) {
    t.join();
  }

  rc = tracker.Shutdown();
  ASSERT_TRUE(rc.OK()) << rc.Report();
  ASSERT_TRUE(fut.get().OK());
}
}  // namespace xgboost::collective
