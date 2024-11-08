/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#pragma once
#include <gtest/gtest.h>
#include <xgboost/global_config.h>  // for InitNewThread

#include <chrono>   // for seconds
#include <cstdint>  // for int32_t
#include <fstream>  // for ifstream
#include <string>   // for string
#include <thread>   // for thread
#include <utility>  // for move
#include <vector>   // for vector

#include "../../../src/collective/comm.h"              // for RabitComm
#include "../../../src/collective/communicator-inl.h"  // for Init, Finalize
#include "../../../src/collective/tracker.h"           // for GetHostAddress
#include "../../../src/common/cuda_rt_utils.h"         // for AllVisibleGPUs
#include "../../../src/common/threading_utils.h"       // for NameThread
#include "../helpers.h"                                // for FileExists

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
  virtual ~WorkerForTest() noexcept(false) { SafeColl(comm_.Shutdown()); }
  auto& Comm() { return comm_; }

  void LimitSockBuf(std::int32_t n_bytes) {
    for (std::int32_t i = 0; i < comm_.World(); ++i) {
      if (i != comm_.Rank()) {
        ASSERT_TRUE(comm_.Chan(i)->Socket()->NonBlocking());
        SafeColl(comm_.Chan(i)->Socket()->SetBufSize(n_bytes));
        SafeColl(comm_.Chan(i)->Socket()->SetNoDelay());
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
    SafeColl(rc);
  }
};

inline Json MakeTrackerConfig(std::string host, std::int32_t n_workers,
                              std::chrono::seconds timeout) {
  Json config{Object{}};
  config["host"] = host;
  config["port"] = Integer{0};
  config["n_workers"] = Integer{n_workers};
  config["sortby"] = Integer{static_cast<std::int32_t>(Tracker::SortBy::kHost)};
  config["timeout"] = static_cast<std::int64_t>(timeout.count());
  return config;
}

template <typename WorkerFn>
void TestDistributed(std::int32_t n_workers, WorkerFn worker_fn,
                     std::chrono::seconds timeout = std::chrono::seconds{2}) {
  std::string host;
  auto rc = GetHostAddress(&host);
  SafeColl(rc);
  LOG(INFO) << "Using " << n_workers << " workers for test.";
  RabitTracker tracker{MakeTrackerConfig(host, n_workers, timeout)};
  auto fut = tracker.Run();

  std::vector<std::thread> workers;
  std::int32_t port = tracker.Port();

  for (std::int32_t i = 0; i < n_workers; ++i) {
    workers.emplace_back([=, init = InitNewThread{}] {
      init();
      worker_fn(host, port, timeout, i);
    });
  }

  for (auto& t : workers) {
    t.join();
  }

  SafeColl(fut.get());
}

inline auto MakeDistributedTestConfig(std::string host, std::int32_t port,
                                      std::chrono::seconds timeout, std::int32_t r) {
  Json config{Object{}};
  config["dmlc_communicator"] = std::string{"rabit"};
  config["dmlc_tracker_uri"] = host;
  config["dmlc_tracker_port"] = port;
  config["dmlc_timeout"] = static_cast<std::int64_t>(timeout.count());
  config["dmlc_task_id"] = std::to_string(r);
  config["dmlc_retry"] = 2;
  return config;
}

template <typename WorkerFn>
void TestDistributedGlobal(std::int32_t n_workers, WorkerFn worker_fn, bool need_finalize = true,
                           std::chrono::seconds test_timeout = std::chrono::seconds{30}) {
  system::SocketStartup();
  std::chrono::seconds poll_timeout{5};

  std::string host;
  auto rc = GetHostAddress(&host);
  SafeColl(rc);

  RabitTracker tracker{MakeTrackerConfig(host, n_workers, poll_timeout)};
  auto fut = tracker.Run();

  std::vector<std::thread> workers;
  std::int32_t port = tracker.Port();

  for (std::int32_t i = 0; i < n_workers; ++i) {
    workers.emplace_back([=, init = InitNewThread{}] {
      init();
      auto fut = std::async(std::launch::async, [=] {
        init();
        auto config = MakeDistributedTestConfig(host, port, poll_timeout, i);
        Init(config);
        worker_fn();
        if (need_finalize) {
          Finalize();
        }
      });
      auto status = fut.wait_for(test_timeout);
      CHECK(status == std::future_status::ready) << "Test timeout";
      fut.get();
    });

    std::string name = "tw-" + std::to_string(i);
    common::NameThread(&workers.back(), name.c_str());
  }

  for (auto& t : workers) {
    t.join();
  }

  SafeColl(fut.get());
  system::SocketFinalize();
}

[[nodiscard]] inline std::int32_t GetWorkerLocalThreads(std::int32_t n_workers) {
  std::int32_t n_total_threads = std::thread::hardware_concurrency();
  auto n_threads = std::max(n_total_threads / n_workers, 1);
  return n_threads;
}

inline void GetWorkerLocalThreads(std::int32_t n_workers, Context* ctx) {
  auto n_threads = GetWorkerLocalThreads(n_workers);
  ctx->UpdateAllowUnknown(
      Args{{"nthread", std::to_string(n_threads)}, {"device", ctx->DeviceName()}});
}

class BaseMGPUTest : public ::testing::Test {
 public:
  /**
   * @param emulate_if_single Emulate multi-GPU for federated test if there's only one GPU
   *                          available.
   */
  template <typename Fn>
  auto DoTest([[maybe_unused]] Fn&& fn, bool is_federated,
              [[maybe_unused]] bool emulate_if_single = false) const {
    auto n_gpus = curt::AllVisibleGPUs();
    if (is_federated) {
#if defined(XGBOOST_USE_FEDERATED)
      if (n_gpus == 1 && emulate_if_single) {
        // Emulate multiple GPUs.
        // We don't use nccl and can have multiple communicators running on the same device.
        n_gpus = 3;
      }
      TestFederatedGlobal(n_gpus, fn);
#else
      GTEST_SKIP_("Not compiled with federated learning.");
#endif  // defined(XGBOOST_USE_FEDERATED)
    } else {
#if defined(XGBOOST_USE_NCCL)
      TestDistributedGlobal(n_gpus, fn);
#else
      GTEST_SKIP_("Not compiled with NCCL.");
#endif  // defined(XGBOOST_USE_NCCL)
    }
  }
};
}  // namespace xgboost::collective
