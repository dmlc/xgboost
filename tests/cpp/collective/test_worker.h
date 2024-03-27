/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#pragma once
#include <gtest/gtest.h>

#include <chrono>   // for seconds
#include <cstdint>  // for int32_t
#include <fstream>  // for ifstream
#include <string>   // for string
#include <thread>   // for thread
#include <utility>  // for move
#include <vector>   // for vector

#include "../../../src/collective/comm.h"
#include "../../../src/collective/tracker.h"  // for GetHostAddress
#include "../helpers.h"                       // for FileExists

#if defined(XGBOOST_USE_FEDERATED)
#include "../plugin/federated/test_worker.h"
#endif  // defined(XGBOOST_USE_FEDERATED)

namespace xgboost::collective {
class WorkerForTest {
  std::string tracker_host_;
  std::int32_t tracker_port_;
  std::int32_t world_size_;

 protected:
  std::int32_t retry_{1};
  std::string task_id_;
  RabitComm comm_;

 public:
  WorkerForTest(std::string host, std::int32_t port, std::chrono::seconds timeout,
                std::int32_t world, std::int32_t rank)
      : tracker_host_{std::move(host)},
        tracker_port_{port},
        world_size_{world},
        task_id_{"t:" + std::to_string(rank)},
        comm_{tracker_host_, tracker_port_, timeout, retry_, task_id_, DefaultNcclName()} {
    CHECK_EQ(world_size_, comm_.World());
  }
  virtual ~WorkerForTest() = default;
  auto& Comm() { return comm_; }

  void LimitSockBuf(std::int32_t n_bytes) {
    for (std::int32_t i = 0; i < comm_.World(); ++i) {
      if (i != comm_.Rank()) {
        ASSERT_TRUE(comm_.Chan(i)->Socket()->NonBlocking());
        ASSERT_TRUE(comm_.Chan(i)->Socket()->SetBufSize(n_bytes).OK());
      }
    }
  }
};

class SocketTest : public ::testing::Test {
 protected:
  std::string skip_msg_{"Skipping IPv6 test"};

  bool SkipTest() {
    std::string path{"/sys/module/ipv6/parameters/disable"};
    if (FileExists(path)) {
      std::ifstream fin(path);
      if (!fin) {
        return true;
      }
      std::string s_value;
      fin >> s_value;
      auto value = std::stoi(s_value);
      if (value != 0) {
        return true;
      }
    } else {
      return true;
    }
    return false;
  }

 protected:
  void SetUp() override { system::SocketStartup(); }
  void TearDown() override { system::SocketFinalize(); }
};

class TrackerTest : public SocketTest {
 public:
  std::int32_t n_workers{2};
  std::chrono::seconds timeout{1};
  std::string host;

  void SetUp() override {
    SocketTest::SetUp();
    auto rc = GetHostAddress(&host);
    ASSERT_TRUE(rc.OK()) << rc.Report();
  }
};

template <typename WorkerFn>
void TestDistributed(std::int32_t n_workers, WorkerFn worker_fn) {
  std::chrono::seconds timeout{2};

  std::string host;
  auto rc = GetHostAddress(&host);
  ASSERT_TRUE(rc.OK()) << rc.Report();
  LOG(INFO) << "Using " << n_workers << " workers for test.";
  RabitTracker tracker{StringView{host}, n_workers, 0, timeout};
  auto fut = tracker.Run();

  std::vector<std::thread> workers;
  std::int32_t port = tracker.Port();

  for (std::int32_t i = 0; i < n_workers; ++i) {
    workers.emplace_back([=] { worker_fn(host, port, timeout, i); });
  }

  for (auto& t : workers) {
    t.join();
  }

  ASSERT_TRUE(fut.get().OK());
}

inline auto MakeDistributedTestConfig(std::string host, std::int32_t port,
                                      std::chrono::seconds timeout, std::int32_t r) {
  Json config{Object{}};
  config["dmlc_communicator"] = std::string{"rabit"};
  config["DMLC_TRACKER_URI"] = host;
  config["DMLC_TRACKER_PORT"] = port;
  config["dmlc_timeout_sec"] = static_cast<std::int64_t>(timeout.count());
  config["DMLC_TASK_ID"] = std::to_string(r);
  config["dmlc_retry"] = 2;
  return config;
}

template <typename WorkerFn>
void TestDistributedGlobal(std::int32_t n_workers, WorkerFn worker_fn, bool need_finalize = true) {
  std::chrono::seconds timeout{1};

  std::string host;
  auto rc = GetHostAddress(&host);
  ASSERT_TRUE(rc.OK()) << rc.Report();
  RabitTracker tracker{StringView{host}, n_workers, 0, timeout};
  auto fut = tracker.Run();

  std::vector<std::thread> workers;
  std::int32_t port = tracker.Port();

  for (std::int32_t i = 0; i < n_workers; ++i) {
    workers.emplace_back([=] {
      auto config = MakeDistributedTestConfig(host, port, timeout, i);
      Init(config);
      worker_fn();
      if (need_finalize) {
        Finalize();
      }
    });
  }

  for (auto& t : workers) {
    t.join();
  }

  ASSERT_TRUE(fut.get().OK());
}

class BaseMGPUTest : public ::testing::Test {
 public:
  template <typename Fn>
  auto DoTest(Fn&& fn, bool is_federated) const {
    auto n_gpus = common::AllVisibleGPUs();
    if (is_federated) {
#if defined(XGBOOST_USE_FEDERATED)
      TestFederatedGlobal(n_gpus, fn);
#endif  // defined(XGBOOST_USE_FEDERATED)
    } else {
      TestDistributedGlobal(n_gpus, fn);
    }
  }
};
}  // namespace xgboost::collective
