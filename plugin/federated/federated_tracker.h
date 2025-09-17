/**
 * Copyright 2022-2023, XGBoost contributors
 */
#pragma once
#include <federated.grpc.pb.h>  // for Server

#include <future>  // for future
#include <memory>  // for unique_ptr
#include <string>  // for string

#include "../../src/collective/in_memory_handler.h"
#include "../../src/collective/tracker.h"  // for Tracker
#include "xgboost/collective/result.h"     // for Result
#include "xgboost/json.h"                  // for Json

namespace xgboost::collective {
namespace federated {
class FederatedService final : public Federated::Service {
 public:
  explicit FederatedService(std::int32_t world_size) : handler_{world_size} {}

  grpc::Status Allgather(grpc::ServerContext* context, AllgatherRequest const* request,
                         AllgatherReply* reply) override;

  grpc::Status AllgatherV(grpc::ServerContext* context, AllgatherVRequest const* request,
                          AllgatherVReply* reply) override;

  grpc::Status Allreduce(grpc::ServerContext* context, AllreduceRequest const* request,
                         AllreduceReply* reply) override;

  grpc::Status Broadcast(grpc::ServerContext* context, BroadcastRequest const* request,
                         BroadcastReply* reply) override;

 private:
  xgboost::collective::InMemoryHandler handler_;
};
};  // namespace federated

class FederatedTracker : public collective::Tracker {
  std::unique_ptr<grpc::Server> server_;
  std::string server_key_path_;
  std::string server_cert_file_;
  std::string client_cert_file_;

 public:
  /**
   * @brief CTOR
   *
   * @param config Configuration, other than the base configuration from Tracker, we have:
   *
   * - federated_secure: bool whether this is a secure server.
   * - server_key_path: path to the key.
   * - server_cert_path: certificate path.
   * - client_cert_path: certificate path for client.
   */
  explicit FederatedTracker(Json const& config);
  ~FederatedTracker() override;
  std::future<Result> Run() override;

  [[nodiscard]] Json WorkerArgs() const override;
  [[nodiscard]] Result Shutdown();
};
}  // namespace xgboost::collective
