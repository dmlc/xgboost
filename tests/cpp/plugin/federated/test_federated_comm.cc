/**
 * Copyright 2022-2024, XGBoost contributors
 */
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>  // for string
#include <thread>  // for thread

#include "../../../../plugin/federated/federated_comm.h"
#include "../../collective/test_worker.h"  // for SocketTest
#include "../../helpers.h"                 // for GMockThrow
#include "test_worker.h"                   // for TestFederated
#include "xgboost/json.h"                  // for Json

namespace xgboost::collective {
namespace {
class FederatedCommTest : public SocketTest {};
auto MakeConfig(std::string host, std::int32_t port, std::int32_t world, std::int32_t rank) {
  Json config{Object{}};
  config["federated_server_address"] = host + ":" + std::to_string(port);
  config["federated_world_size"] = Integer{world};
  config["federated_rank"] = Integer{rank};
  return config;
}
}  // namespace

TEST_F(FederatedCommTest, ThrowOnWorldSizeTooSmall) {
  auto config = MakeConfig("localhost", 0, 0, 0);
  auto construct = [config] {
    FederatedComm comm{DefaultRetry(), std::chrono::seconds{DefaultTimeoutSec()}, "", config};
  };
  ASSERT_THAT(construct, GMockThrow("Invalid world size"));
}

TEST_F(FederatedCommTest, ThrowOnRankTooSmall) {
  auto config = MakeConfig("localhost", 0, 1, -1);
  auto construct = [config] {
    FederatedComm comm{DefaultRetry(), std::chrono::seconds{DefaultTimeoutSec()}, "", config};
  };
  ASSERT_THAT(construct, GMockThrow("Invalid worker rank."));
}

TEST_F(FederatedCommTest, ThrowOnRankTooBig) {
  auto config = MakeConfig("localhost", 0, 1, 1);
  auto construct = [config] {
    FederatedComm comm{DefaultRetry(), std::chrono::seconds{DefaultTimeoutSec()}, "", config};
  };
  ASSERT_THAT(construct, GMockThrow("Invalid worker rank."));
}

TEST_F(FederatedCommTest, ThrowOnWorldSizeNotInteger) {
  auto construct = [] {
    Json config{Object{}};
    config["federated_server_address"] = std::string("localhost:0");
    config["federated_world_size"] = std::string("1");
    config["federated_rank"] = Integer(0);
    FederatedComm comm{DefaultRetry(), std::chrono::seconds{DefaultTimeoutSec()}, "", config};
  };
  ASSERT_THAT(construct, GMockThrow("got: `String`"));
}

TEST_F(FederatedCommTest, ThrowOnRankNotInteger) {
  auto construct = [] {
    Json config{Object{}};
    config["federated_server_address"] = std::string("localhost:0");
    config["federated_world_size"] = 1;
    config["federated_rank"] = std::string("0");
    FederatedComm comm(DefaultRetry(), std::chrono::seconds{DefaultTimeoutSec()}, "", config);
  };
  ASSERT_THAT(construct, GMockThrow("got: `String`"));
}

TEST_F(FederatedCommTest, GetWorldSizeAndRank) {
  Json config{Object{}};
  config["federated_world_size"] = 6;
  config["federated_rank"] = 3;
  config["federated_server_address"] = String{"localhost:0"};
  FederatedComm comm{DefaultRetry(), std::chrono::seconds{DefaultTimeoutSec()}, "", config};
  EXPECT_EQ(comm.World(), 6);
  EXPECT_EQ(comm.Rank(), 3);
}

TEST_F(FederatedCommTest, IsDistributed) {
  FederatedComm comm{DefaultRetry(), std::chrono::seconds{DefaultTimeoutSec()}, "",
                     MakeConfig("localhost", 0, 2, 1)};
  EXPECT_TRUE(comm.IsDistributed());
}

TEST_F(FederatedCommTest, InsecureTracker) {
  std::int32_t n_workers = std::min(std::thread::hardware_concurrency(), 3u);
  TestFederated(n_workers, [=](std::shared_ptr<FederatedComm> comm, std::int32_t rank) {
    ASSERT_EQ(comm->Rank(), rank);
    ASSERT_EQ(comm->World(), n_workers);
  });
}
}  // namespace xgboost::collective
