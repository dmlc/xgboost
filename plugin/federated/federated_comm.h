/**
 * Copyright 2023-2024, XGBoost contributors
 */
#pragma once

#include <federated.grpc.pb.h>
#include <federated.pb.h>

#include <chrono>   // for seconds
#include <cstdint>  // for int32_t
#include <memory>   // for shared_ptr
#include <string>   // for string

#include "../../src/collective/comm.h"    // for HostComm
#include "xgboost/json.h"

namespace xgboost::collective {
class FederatedComm : public HostComm {
  std::shared_ptr<federated::Federated::Stub> stub_;

  void Init(std::string const& host, std::int32_t port, std::int32_t world, std::int32_t rank,
            std::string const& server_cert, std::string const& client_key,
            std::string const& client_cert);

 protected:
  explicit FederatedComm(std::shared_ptr<FederatedComm const> that) : stub_{that->stub_} {
    this->rank_ = that->Rank();
    this->world_ = that->World();

    this->retry_ = that->Retry();
    this->timeout_ = that->Timeout();
    this->task_id_ = that->TaskID();

    this->tracker_ = that->TrackerInfo();
  }

 public:
  /**
   * @param config
   *
   * - federated_server_address: Tracker address
   * - federated_world_size: The number of workers
   * - federated_rank: Rank of federated worker
   * - federated_server_cert_path
   * - federated_client_key_path
   * - federated_client_cert_path
   */
  explicit FederatedComm(std::int32_t retry, std::chrono::seconds timeout, std::string task_id,
                         Json const& config);
  [[nodiscard]] Result Shutdown() final {
    this->ResetState();
    return Success();
  }
  ~FederatedComm() override { stub_.reset(); }

  [[nodiscard]] std::shared_ptr<Channel> Chan(std::int32_t) const override {
    LOG(FATAL) << "peer to peer communication is not allowed for federated learning.";
    return nullptr;
  }
  [[nodiscard]] Result LogTracker(std::string msg) const override {
    LOG(CONSOLE) << msg;
    return Success();
  }
  [[nodiscard]] bool IsFederated() const override { return true; }
  [[nodiscard]] federated::Federated::Stub* Handle() const { return stub_.get(); }

  [[nodiscard]] Comm* MakeCUDAVar(Context const* ctx, std::shared_ptr<Coll> pimpl) const override;
  /**
   * @brief Get a string ID for the current process.
   */
  [[nodiscard]] Result ProcessorName(std::string* out) const final {
    auto rank = this->Rank();
    *out = "rank:" + std::to_string(rank);
    return Success();
  };
};
}  // namespace xgboost::collective
