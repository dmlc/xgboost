/**
 * Copyright 2023, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include "../../../src/collective/comm.h"
#include "test_worker.h"
namespace xgboost::collective {
namespace {
class CommTest : public TrackerTest {};
}  // namespace

TEST_F(CommTest, Channel) {
  auto n_workers = 4;
  RabitTracker tracker{host, n_workers, 0, timeout};
  auto fut = tracker.Run();

  std::vector<std::thread> workers;
  std::int32_t port = tracker.Port();

  for (std::int32_t i = 0; i < n_workers; ++i) {
    workers.emplace_back([=] {
      WorkerForTest worker{host, port, timeout, n_workers, i};
      if (i % 2 == 0) {
        auto p_chan = worker.Comm().Chan(i + 1);
        p_chan->SendAll(
            EraseType(common::Span<std::int32_t const>{&i, static_cast<std::size_t>(1)}));
        auto rc = p_chan->Block();
        ASSERT_TRUE(rc.OK()) << rc.Report();
      } else {
        auto p_chan = worker.Comm().Chan(i - 1);
        std::int32_t r{-1};
        p_chan->RecvAll(EraseType(common::Span<std::int32_t>{&r, static_cast<std::size_t>(1)}));
        auto rc = p_chan->Block();
        ASSERT_TRUE(rc.OK()) << rc.Report();
        ASSERT_EQ(r, i - 1);
      }
    });
  }

  for (auto &w : workers) {
    w.join();
  }

  ASSERT_TRUE(fut.get().OK());
}
}  // namespace xgboost::collective
