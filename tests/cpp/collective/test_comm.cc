/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include "../../../src/collective/comm.h"
#include "../../../src/common/type.h"  // for EraseType
#include "test_worker.h"               // for TrackerTest

namespace xgboost::collective {
namespace {
class CommTest : public TrackerTest {};
}  // namespace

TEST_F(CommTest, Channel) {
  auto n_workers = 4;
  RabitTracker tracker{MakeTrackerConfig(host, n_workers, timeout)};
  auto fut = tracker.Run();

  std::vector<std::thread> workers;
  std::int32_t port = tracker.Port();

  for (std::int32_t i = 0; i < n_workers; ++i) {
    workers.emplace_back([=] {
      WorkerForTest worker{host, port, timeout, n_workers, i};
      if (i % 2 == 0) {
        auto p_chan = worker.Comm().Chan(i + 1);
        auto rc = Success() << [&] {
          return p_chan->SendAll(
              EraseType(common::Span<std::int32_t const>{&i, static_cast<std::size_t>(1)}));
        } << [&] { return p_chan->Block(); };
        SafeColl(rc);
      } else {
        auto p_chan = worker.Comm().Chan(i - 1);
        std::int32_t r{-1};
        auto rc = Success() << [&] {
          return p_chan->RecvAll(
              EraseType(common::Span<std::int32_t>{&r, static_cast<std::size_t>(1)}));
        } << [&] { return p_chan->Block(); };
        SafeColl(rc);
        ASSERT_EQ(r, i - 1);
      }
    });
  }

  for (auto &w : workers) {
    w.join();
  }

  SafeColl(fut.get());
}
}  // namespace xgboost::collective
