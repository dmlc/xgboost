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
 * @brief A federated learning communicator class that handles collective communication.
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
                 Operation op) override {
    std::string const send_buffer(reinterpret_cast<char const *>(send_receive_buffer),
                                  count * GetTypeSize(data_type));
    auto const received =
        client_->Allreduce(send_buffer, ConvertDataType(data_type), ConvertOperation(op));
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

 private:
  static xgboost::federated::DataType ConvertDataType(DataType data_type) {
    xgboost::federated::DataType result{};
    switch (data_type) {
      case DataType::kInt:
        result = xgboost::federated::DataType::INT;
        break;
      case DataType::kFloat:
        result = xgboost::federated::DataType::FLOAT;
        break;
      case DataType::kDouble:
        result = xgboost::federated::DataType::DOUBLE;
        break;
    }
    return result;
  }

  static xgboost::federated::ReduceOperation ConvertOperation(Operation operation) {
    xgboost::federated::ReduceOperation result{};
    switch (operation) {
      case Operation::kMax:
        result = xgboost::federated::ReduceOperation::MAX;
        break;
      case Operation::kSum:
        result = xgboost::federated::ReduceOperation::SUM;
        break;
    }
    return result;
  }

  std::unique_ptr<xgboost::federated::FederatedClient> client_{};
};

}  // namespace collective
}  // namespace xgboost
