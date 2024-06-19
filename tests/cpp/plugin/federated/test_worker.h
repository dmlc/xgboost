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
#include "../../../../src/collective/communicator-inl.h"
#include "federated_comm.h"  // for FederatedComm
#include "xgboost/json.h"    // for Json

namespace xgboost::collective {
inline Json FederatedTestConfig(std::int32_t n_workers, std::int32_t port, std::int32_t i) {
  Json config{Object{}};
  config["dmlc_communicator"] = std::string{"federated"};
  config["dmlc_task_id"] = std::to_string(i);
  config["dmlc_retry"] = 2;
  config["federated_world_size"] = n_workers;
  config["federated_rank"] = i;
  config["federated_server_address"] = "0.0.0.0:" + std::to_string(port);
  return config;
}

template <typename WorkerFn>
void TestFederatedImpl(std::int32_t n_workers, WorkerFn&& fn) {
  Json config{Object()};
  config["federated_secure"] = Boolean{false};
  config["n_workers"] = Integer{n_workers};
  FederatedTracker tracker{config};
  auto fut = tracker.Run();

  std::vector<std::thread> workers;
  using namespace std::chrono_literals;
  auto rc = tracker.WaitUntilReady();
  SafeColl(rc);
  std::int32_t port = tracker.Port();

  for (std::int32_t i = 0; i < n_workers; ++i) {
    workers.emplace_back([=] { fn(port, i); });
  }

  for (auto& t : workers) {
    t.join();
  }

  rc = tracker.Shutdown();
  SafeColl(rc);
  SafeColl(fut.get());
}

template <typename WorkerFn>
void TestFederated(std::int32_t n_workers, WorkerFn&& fn) {
  TestFederatedImpl(n_workers, [&](std::int32_t port, std::int32_t i) {
    auto config = FederatedTestConfig(n_workers, port, i);
    auto comm = std::make_shared<FederatedComm>(
        DefaultRetry(), std::chrono::seconds{DefaultTimeoutSec()}, std::to_string(i), config);

    fn(comm, i);
  });
}

template <typename WorkerFn>
void TestFederatedGroup(std::int32_t n_workers, WorkerFn&& fn) {
  TestFederatedImpl(n_workers, [&](std::int32_t port, std::int32_t i) {
    auto config = FederatedTestConfig(n_workers, port, i);
    std::shared_ptr<CommGroup> comm_group{CommGroup::Create(config)};
    fn(comm_group, i);
  });
}

template <typename WorkerFn>
void TestFederatedGlobal(std::int32_t n_workers, WorkerFn&& fn) {
  TestFederatedImpl(n_workers, [&](std::int32_t port, std::int32_t i) {
    auto config = FederatedTestConfig(n_workers, port, i);
    collective::Init(config);
    fn();
    collective::Finalize();
  });
}
}  // namespace xgboost::collective
