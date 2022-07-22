/*!
 * Copyright 2022 XGBoost contributors
 */
#include <grpcpp/server_builder.h>
#include <gtest/gtest.h>

#include <thread>

#include "../../../plugin/federated/federated_communicator.h"
#include "../../../plugin/federated/federated_server.h"

namespace xgboost {
namespace collective {

std::string const kServerAddress{"localhost:56789"};  // NOLINT(cert-err58-cpp)

class FederatedCommunicatorTest : public ::testing::Test {
 public:
  static void VerifyAllreduce(int rank) {
    FederatedCommunicator comm{kWorldSize, rank, kServerAddress};
    CheckAllreduce(comm);
  }

  static void VerifyBroadcast(int rank) {
    FederatedCommunicator comm{kWorldSize, rank, kServerAddress};
    CheckBroadcast(comm, rank);
  }

 protected:
  void SetUp() override {
    server_thread_.reset(new std::thread([this] {
      grpc::ServerBuilder builder;
      federated::FederatedService service{kWorldSize};
      builder.AddListeningPort(kServerAddress, grpc::InsecureServerCredentials());
      builder.RegisterService(&service);
      server_ = builder.BuildAndStart();
      server_->Wait();
    }));
  }

  void TearDown() override {
    server_->Shutdown();
    server_thread_->join();
  }

  static void CheckAllreduce(FederatedCommunicator& comm) {
    int buffer[] = {1, 2, 3, 4, 5};
    comm.AllReduce(buffer, sizeof(buffer) / sizeof(buffer[0]), DataType::kInt32, Operation::kSum);
    int expected[] = {3, 6, 9, 12, 15};
    for (auto i = 0; i < 5; i++) {
      EXPECT_EQ(buffer[i], expected[i]);
    }
  }

  static void CheckBroadcast(FederatedCommunicator& comm, int rank) {
    if (rank == 0) {
      std::string buffer{"hello"};
      comm.Broadcast(&buffer[0], buffer.size(), 0);
      EXPECT_EQ(buffer, "hello");
    } else {
      std::string buffer{"     "};
      comm.Broadcast(&buffer[0], buffer.size(), 0);
      EXPECT_EQ(buffer, "hello");
    }
  }

  static int const kWorldSize{3};
  std::unique_ptr<std::thread> server_thread_;
  std::unique_ptr<grpc::Server> server_;
};

TEST(FederatedCommunicatorSimpleTest, ThrowOnWorldSizeTooSmall) {
  auto construct = []() { FederatedCommunicator comm{0, 0, kServerAddress, "", "", ""}; };
  EXPECT_THROW(construct(), dmlc::Error);
}

TEST(FederatedCommunicatorSimpleTest, ThrowOnRankTooSmall) {
  auto construct = []() { FederatedCommunicator comm{1, -1, kServerAddress, "", "", ""}; };
  EXPECT_THROW(construct(), dmlc::Error);
}

TEST(FederatedCommunicatorSimpleTest, ThrowOnRankTooBig) {
  auto construct = []() { FederatedCommunicator comm{1, 1, kServerAddress, "", "", ""}; };
  EXPECT_THROW(construct(), dmlc::Error);
}

TEST(FederatedCommunicatorSimpleTest, GetWorldSizeAndRank) {
  FederatedCommunicator comm{6, 3, kServerAddress};
  EXPECT_EQ(comm.GetWorldSize(), 6);
  EXPECT_EQ(comm.GetRank(), 3);
}

TEST(FederatedCommunicatorSimpleTest, IsNotDistributed) {
  FederatedCommunicator comm{1, 0, kServerAddress};
  EXPECT_FALSE(comm.IsDistributed());
}

TEST(FederatedCommunicatorSimpleTest, IsDistributed) {
  FederatedCommunicator comm{2, 1, kServerAddress};
  EXPECT_TRUE(comm.IsDistributed());
}

TEST_F(FederatedCommunicatorTest, Allreduce) {
  std::vector<std::thread> threads;
  for (auto rank = 0; rank < kWorldSize; rank++) {
    threads.emplace_back(std::thread(&FederatedCommunicatorTest::VerifyAllreduce, rank));
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

TEST_F(FederatedCommunicatorTest, Broadcast) {
  std::vector<std::thread> threads;
  for (auto rank = 0; rank < kWorldSize; rank++) {
    threads.emplace_back(std::thread(&FederatedCommunicatorTest::VerifyBroadcast, rank));
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

TEST(FederatedCommunicatorFactoryTest, ServerAddress) {
    FederatedCommunicatorFactory factory0{0, nullptr};
    EXPECT_EQ(factory0.GetServerAddress(), "");

    setenv("FEDERATED_SERVER_ADDRESS", "localhost:9091", 1);
    FederatedCommunicatorFactory factory1{0, nullptr};
    EXPECT_EQ(factory1.GetServerAddress(), "localhost:9091");

    char *args[1];
    args[0] = strdup("federated_server_address=foo:9091");
    FederatedCommunicatorFactory factory2{1, args};
    EXPECT_EQ(factory2.GetServerAddress(), "foo:9091");
}

TEST(FederatedCommunicatorFactoryTest, WorldSize) {
    FederatedCommunicatorFactory factory0{0, nullptr};
    EXPECT_EQ(factory0.GetWorldSize(), 0);

    setenv("FEDERATED_WORLD_SIZE", "2", 1);
    FederatedCommunicatorFactory factory1{0, nullptr};
    EXPECT_EQ(factory1.GetWorldSize(), 2);

    char *args[1];
    args[0] = strdup("federated_world_size=3");
    FederatedCommunicatorFactory factory2{1, args};
    EXPECT_EQ(factory2.GetWorldSize(), 3);
}

TEST(FederatedCommunicatorFactoryTest, Rank) {
    FederatedCommunicatorFactory factory0{0, nullptr};
    EXPECT_EQ(factory0.GetRank(), -1);

    setenv("FEDERATED_RANK", "1", 1);
    FederatedCommunicatorFactory factory1{0, nullptr};
    EXPECT_EQ(factory1.GetRank(), 1);

    char *args[1];
    args[0] = strdup("federated_rank=2");
    FederatedCommunicatorFactory factory2{1, args};
    EXPECT_EQ(factory2.GetRank(), 2);
}

TEST(FederatedCommunicatorFactoryTest, ServerCert) {
    FederatedCommunicatorFactory factory0{0, nullptr};
    EXPECT_EQ(factory0.GetServerCert(), "");

    setenv("FEDERATED_SERVER_CERT", "foo/server-cert.pem", 1);
    FederatedCommunicatorFactory factory1{0, nullptr};
    EXPECT_EQ(factory1.GetServerCert(), "foo/server-cert.pem");

    char *args[1];
    args[0] = strdup("federated_server_cert=bar/server-cert.pem");
    FederatedCommunicatorFactory factory2{1, args};
    EXPECT_EQ(factory2.GetServerCert(), "bar/server-cert.pem");
}

TEST(FederatedCommunicatorFactoryTest, ClientKey) {
    FederatedCommunicatorFactory factory0{0, nullptr};
    EXPECT_EQ(factory0.GetClientKey(), "");

    setenv("FEDERATED_CLIENT_KEY", "foo/client-key.pem", 1);
    FederatedCommunicatorFactory factory1{0, nullptr};
    EXPECT_EQ(factory1.GetClientKey(), "foo/client-key.pem");

    char *args[1];
    args[0] = strdup("federated_client_key=bar/client-key.pem");
    FederatedCommunicatorFactory factory2{1, args};
    EXPECT_EQ(factory2.GetClientKey(), "bar/client-key.pem");
}

TEST(FederatedCommunicatorFactoryTest, ClientCert) {
    FederatedCommunicatorFactory factory0{0, nullptr};
    EXPECT_EQ(factory0.GetClientCert(), "");

    setenv("FEDERATED_CLIENT_CERT", "foo/client-cert.pem", 1);
    FederatedCommunicatorFactory factory1{0, nullptr};
    EXPECT_EQ(factory1.GetClientCert(), "foo/client-cert.pem");

    char *args[1];
    args[0] = strdup("federated_client_cert=bar/client-cert.pem");
    FederatedCommunicatorFactory factory2{1, args};
    EXPECT_EQ(factory2.GetClientCert(), "bar/client-cert.pem");
}

}  // namespace collective
}  // namespace xgboost
