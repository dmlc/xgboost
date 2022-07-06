/*!
 * Copyright 2022 XGBoost contributors
 */
#pragma once
#include "../../src/collective/communicator.h"
#include "../../src/common/io.h"
#include "federated_client.h"

namespace xgboost {
namespace collective {

/**
 * @brief A federated learning communicator class that handles collective communication .
 */
class FederatedCommunicator : public Communicator {
 public:
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
    client_.reset(new xgboost::federated::FederatedClient(
        server_address, rank, xgboost::common::ReadAll(server_cert_path),
        xgboost::common::ReadAll(client_key_path), xgboost::common::ReadAll(client_cert_path)));
  }

  /** @brief Insecure communicator for testing only. */
  FederatedCommunicator(int world_size, int rank, std::string const &server_address)
      : Communicator{world_size, rank} {
    client_.reset(new xgboost::federated::FederatedClient(server_address, rank));
  }

  ~FederatedCommunicator() override { client_.reset(); }

  void AllReduce(void *send_receive_buffer, std::size_t count, DataType data_type,
                 Operation op) override {}

  void Broadcast(void *send_receive_buffer, std::size_t size, int root) override {
    if (GetWorldSize() == 1) return;
    if (GetRank() == root) {
      std::string const send_buffer(reinterpret_cast<char *>(send_receive_buffer), size);
      client_->Broadcast(send_buffer, root);
    } else {
      auto const received = client_->Broadcast("", root);
      received.copy(reinterpret_cast<char *>(send_receive_buffer), size);
    }
  }

 private:
  std::unique_ptr<xgboost::federated::FederatedClient> client_{};
};

}  // namespace collective
}  // namespace xgboost
