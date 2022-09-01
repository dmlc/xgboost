/*!
 * Copyright 2022 XGBoost contributors
 */
#pragma once
#include <xgboost/json.h>

#include "../../src/collective/communicator.h"
#include "../../src/common/io.h"
#include "federated_client.h"

namespace xgboost {
namespace collective {

/**
 * @brief A federated learning communicator class that handles collective communication.
 */
class FederatedCommunicator : public Communicator {
 public:
  static Communicator *Create(Json const &config) {
    std::string server_address{};
    int world_size{0};
    int rank{-1};
    std::string server_cert{};
    std::string client_key{};
    std::string client_cert{};

    // Parse environment variables first.
    auto *value = getenv("FEDERATED_SERVER_ADDRESS");
    if (value != nullptr) {
      server_address = value;
    }
    value = getenv("FEDERATED_WORLD_SIZE");
    if (value != nullptr) {
      world_size = std::stoi(value);
    }
    value = getenv("FEDERATED_RANK");
    if (value != nullptr) {
      rank = std::stoi(value);
    }
    value = getenv("FEDERATED_SERVER_CERT");
    if (value != nullptr) {
      server_cert = value;
    }
    value = getenv("FEDERATED_CLIENT_KEY");
    if (value != nullptr) {
      client_key = value;
    }
    value = getenv("FEDERATED_CLIENT_CERT");
    if (value != nullptr) {
      client_cert = value;
    }

    // Runtime configuration overrides.
    auto const &j_server_address = config["federated_server_address"];
    if (IsA<String const>(j_server_address)) {
      server_address = get<String const>(j_server_address);
    }
    auto const &j_world_size = config["federated_world_size"];
    if (IsA<Integer const>(j_world_size)) {
      world_size = static_cast<int>(get<Integer const>(j_world_size));
    }
    auto const &j_rank = config["federated_rank"];
    if (IsA<Integer const>(j_rank)) {
      rank = static_cast<int>(get<Integer const>(j_rank));
    }
    auto const &j_server_cert = config["federated_server_cert"];
    if (IsA<String const>(j_server_cert)) {
      server_cert = get<String const>(j_server_cert);
    }
    auto const &j_client_key = config["federated_client_key"];
    if (IsA<String const>(j_client_key)) {
      client_key = get<String const>(j_client_key);
    }
    auto const &j_client_cert = config["federated_client_cert"];
    if (IsA<String const>(j_client_cert)) {
      client_cert = get<String const>(j_client_cert);
    }

    if (server_address.empty()) {
      LOG(FATAL) << "Federated server address must be set.";
    }
    if (world_size == 0) {
      LOG(FATAL) << "Federated world size must be set.";
    }
    if (rank == -1) {
      LOG(FATAL) << "Federated rank must be set.";
    }
    return new FederatedCommunicator(world_size, rank, server_address, server_cert, client_key,
                                     client_cert);
  }

  /**
   * @brief Construct a new federated communicator.
   *
   * @param world_size Total number of processes.
   * @param rank       Rank of the current process.
   */
  FederatedCommunicator(int world_size, int rank, std::string const &server_address,
                        std::string const &server_cert_path, std::string const &client_key_path,
                        std::string const &client_cert_path)
      : Communicator{world_size, rank} {
    if (server_cert_path.empty() || client_key_path.empty() || client_cert_path.empty()) {
      client_.reset(new xgboost::federated::FederatedClient(server_address, rank));
    } else {
      client_.reset(new xgboost::federated::FederatedClient(
          server_address, rank, xgboost::common::ReadAll(server_cert_path),
          xgboost::common::ReadAll(client_key_path), xgboost::common::ReadAll(client_cert_path)));
    }
  }

  /** @brief Insecure communicator for testing only. */
  FederatedCommunicator(int world_size, int rank, std::string const &server_address)
      : Communicator{world_size, rank} {
    client_.reset(new xgboost::federated::FederatedClient(server_address, rank));
  }

  ~FederatedCommunicator() override { client_.reset(); }

  bool IsDistributed() const override { return true; }

  void AllReduce(void *send_receive_buffer, std::size_t count, DataType data_type,
                 Operation op) override {
    std::string const send_buffer(reinterpret_cast<char const *>(send_receive_buffer),
                                  count * GetTypeSize(data_type));
    auto const received =
        client_->Allreduce(send_buffer, static_cast<xgboost::federated::DataType>(data_type),
                           static_cast<xgboost::federated::ReduceOperation>(op));
    received.copy(reinterpret_cast<char *>(send_receive_buffer), count * GetTypeSize(data_type));
  }

  void Broadcast(void *send_receive_buffer, std::size_t size, int root) override {
    if (GetWorldSize() == 1) return;
    if (GetRank() == root) {
      std::string const send_buffer(reinterpret_cast<char const *>(send_receive_buffer), size);
      client_->Broadcast(send_buffer, root);
    } else {
      auto const received = client_->Broadcast("", root);
      received.copy(reinterpret_cast<char *>(send_receive_buffer), size);
    }
  }

  std::string GetProcessorName() override { return "rank" + std::to_string(GetRank()); }

  void Print(const std::string &message) override { LOG(CONSOLE) << message; }

 private:
  std::unique_ptr<xgboost::federated::FederatedClient> client_{};
};
}  // namespace collective
}  // namespace xgboost
