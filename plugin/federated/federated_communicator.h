/*!
 * Copyright 2022 XGBoost contributors
 */
#pragma once
#include <xgboost/json.h>

#include "../../src/c_api/c_api_utils.h"
#include "../../src/collective/communicator.h"
#include "../../src/common/io.h"
#include "federated_client.h"

namespace xgboost {
namespace collective {

/**
 * @brief A Federated Learning communicator class that handles collective communication.
 */
class FederatedCommunicator : public Communicator {
 public:
  /**
   * @brief Create a new communicator based on JSON configuration.
   * @param config JSON configuration.
   * @return Communicator as specified by the JSON configuration.
   */
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

    // Runtime configuration overrides, optional as users can specify them as env vars.
    server_address = OptionalArg<String>(config, "federated_server_address", server_address);
    world_size =
        OptionalArg<Integer>(config, "federated_world_size", static_cast<Integer::Int>(world_size));
    rank = OptionalArg<Integer>(config, "federated_rank", static_cast<Integer::Int>(rank));
    server_cert = OptionalArg<String>(config, "federated_server_cert", server_cert);
    client_key = OptionalArg<String>(config, "federated_client_key", client_key);
    client_cert = OptionalArg<String>(config, "federated_client_cert", client_cert);

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
   * @param world_size       Total number of processes.
   * @param rank             Rank of the current process.
   * @param server_address   Address of the federated server (host:port).
   * @param server_cert_path Path to the server cert file.
   * @param client_key_path  Path to the client key file.
   * @param client_cert_path Path to the client cert file.
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

  /**
   * @brief Construct an insecure federated communicator without using SSL.
   * @param world_size     Total number of processes.
   * @param rank           Rank of the current process.
   * @param server_address Address of the federated server (host:port).
   */
  FederatedCommunicator(int world_size, int rank, std::string const &server_address)
      : Communicator{world_size, rank} {
    client_.reset(new xgboost::federated::FederatedClient(server_address, rank));
  }

  ~FederatedCommunicator() override { client_.reset(); }

  /**
   * \brief Get if the communicator is distributed.
   * \return True.
   */
  bool IsDistributed() const override { return true; }

  /**
   * \brief Get if the communicator is federated.
   * \return True.
   */
  bool IsFederated() const override { return true; }

  /**
   * \brief Perform in-place allgather.
   * \param send_receive_buffer Buffer for both sending and receiving data.
   * \param size Number of bytes to be gathered.
   */
  void AllGather(void *send_receive_buffer, std::size_t size) override {
    std::string const send_buffer(reinterpret_cast<char const *>(send_receive_buffer), size);
    auto const received = client_->Allgather(send_buffer);
    received.copy(reinterpret_cast<char *>(send_receive_buffer), size);
  }

  /**
   * \brief Perform in-place allreduce.
   * \param send_receive_buffer Buffer for both sending and receiving data.
   * \param count Number of elements to be reduced.
   * \param data_type Enumeration of data type.
   * \param op Enumeration of operation type.
   */
  void AllReduce(void *send_receive_buffer, std::size_t count, DataType data_type,
                 Operation op) override {
    std::string const send_buffer(reinterpret_cast<char const *>(send_receive_buffer),
                                  count * GetTypeSize(data_type));
    auto const received =
        client_->Allreduce(send_buffer, static_cast<xgboost::federated::DataType>(data_type),
                           static_cast<xgboost::federated::ReduceOperation>(op));
    received.copy(reinterpret_cast<char *>(send_receive_buffer), count * GetTypeSize(data_type));
  }

  /**
   * \brief Broadcast a memory region to all others from root.
   * \param send_receive_buffer Pointer to the send or receive buffer.
   * \param size Size of the data.
   * \param root The process rank to broadcast from.
   */
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

  /**
   * \brief Get the name of the processor.
   * \return Name of the processor.
   */
  std::string GetProcessorName() override { return "rank" + std::to_string(GetRank()); }

  /**
   * \brief Print the message to the communicator.
   * \param message The message to be printed.
   */
  void Print(const std::string &message) override { LOG(CONSOLE) << message; }

 protected:
  void Shutdown() override {}

 private:
  std::unique_ptr<xgboost::federated::FederatedClient> client_{};
};
}  // namespace collective
}  // namespace xgboost
