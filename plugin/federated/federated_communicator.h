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

  std::string GetProcessorName() override { return "rank" + std::to_string(GetRank()); }

  void Print(const std::string &message) override { LOG(CONSOLE) << message; }

 private:
  static xgboost::federated::DataType ConvertDataType(DataType data_type) {
    xgboost::federated::DataType result{};
    switch (data_type) {
      case DataType::kInt8:
        result = xgboost::federated::DataType::CHAR;
        break;
      case DataType::kUInt8:
        result = xgboost::federated::DataType::UCHAR;
        break;
      case DataType::kInt32:
        result = xgboost::federated::DataType::INT;
        break;
      case DataType::kUInt32:
        result = xgboost::federated::DataType::UINT;
        break;
      case DataType::kInt64:
        result = xgboost::federated::DataType::LONG;
        break;
      case DataType::kUInt64:
        result = xgboost::federated::DataType::ULONG;
        break;
      case DataType::kFloat:
        result = xgboost::federated::DataType::FLOAT;
        break;
      case DataType::kDouble:
        result = xgboost::federated::DataType::DOUBLE;
        break;
      default:
        LOG(FATAL) << "Unknown data type.";
    }
    return result;
  }

  static xgboost::federated::ReduceOperation ConvertOperation(Operation operation) {
    xgboost::federated::ReduceOperation result{};
    switch (operation) {
      case Operation::kMax:
        result = xgboost::federated::ReduceOperation::MAX;
        break;
      case Operation::kMin:
        result = xgboost::federated::ReduceOperation::MIN;
        break;
      case Operation::kSum:
        result = xgboost::federated::ReduceOperation::SUM;
        break;
      default:
        LOG(FATAL) << "Unknown reduce operation.";
    }
    return result;
  }

  std::unique_ptr<xgboost::federated::FederatedClient> client_{};
};

class FederatedCommunicatorFactory {
 public:
  FederatedCommunicatorFactory(int argc, char *argv[]) {
    // Parse environment variables first.
    for (auto const &env_var : env_vars_) {
      char const *value = getenv(env_var.c_str());
      if (value != nullptr) {
        SetParam(env_var, value);
      }
    }

    // Command line argument overrides.
    for (int i = 0; i < argc; ++i) {
      std::string const key_value = argv[i];
      auto const delimiter = key_value.find('=');
      if (delimiter != std::string::npos) {
        SetParam(key_value.substr(0, delimiter), key_value.substr(delimiter + 1));
      }
    }
  }

  Communicator *Create() {
    if (server_address_.empty()) {
      LOG(FATAL) << "Federated server address must be set.";
    }
    if (world_size_ == 0) {
      LOG(FATAL) << "Federated world size must be set.";
    }
    if (rank_ == -1) {
      LOG(FATAL) << "Federated rank must be set.";
    }
    if (server_cert_.empty()) {
      LOG(FATAL) << "Federated server cert must be set.";
    }
    if (client_key_.empty()) {
      LOG(FATAL) << "Federated client key must be set.";
    }
    if (client_cert_.empty()) {
      LOG(FATAL) << "Federated client cert must be set.";
    }
    return new FederatedCommunicator(world_size_, rank_, server_address_, server_cert_, client_key_,
                                     client_cert_);
  }

  std::string const &GetServerAddress() const { return server_address_; }
  int GetWorldSize() const { return world_size_; }
  int GetRank() const { return rank_; }
  std::string const &GetServerCert() const { return server_cert_; }
  std::string const &GetClientKey() const { return client_key_; }
  std::string const &GetClientCert() const { return client_cert_; }

 private:
  void SetParam(std::string const &name, std::string const &val) {
    if (!strcasecmp(name.c_str(), "FEDERATED_SERVER_ADDRESS")) {
      server_address_ = val;
    } else if (!strcasecmp(name.c_str(), "FEDERATED_WORLD_SIZE")) {
      world_size_ = std::stoi(val);
    } else if (!strcasecmp(name.c_str(), "FEDERATED_RANK")) {
      rank_ = std::stoi(val);
    } else if (!strcasecmp(name.c_str(), "FEDERATED_SERVER_CERT")) {
      server_cert_ = val;
    } else if (!strcasecmp(name.c_str(), "FEDERATED_CLIENT_KEY")) {
      client_key_ = val;
    } else if (!strcasecmp(name.c_str(), "FEDERATED_CLIENT_CERT")) {
      client_cert_ = val;
    }
  }

  // clang-format off
  std::vector<std::string> const env_vars_{
      "FEDERATED_SERVER_ADDRESS",
      "FEDERATED_WORLD_SIZE",
      "FEDERATED_RANK",
      "FEDERATED_SERVER_CERT",
      "FEDERATED_CLIENT_KEY",
      "FEDERATED_CLIENT_CERT" };
  // clang-format on

  std::string server_address_{};
  int world_size_{0};
  int rank_{-1};
  std::string server_cert_{};
  std::string client_key_{};
  std::string client_cert_{};
};

}  // namespace collective
}  // namespace xgboost
