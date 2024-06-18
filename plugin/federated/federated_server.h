/**
 * Copyright 2022-2023, XGBoost contributors
 */
#pragma once

#include <federated.old.grpc.pb.h>

#include <cstdint>  // for int32_t
#include <future>   // for future

#include "../../src/collective/in_memory_handler.h"
#include "../../src/collective/tracker.h"  // for Tracker
#include "xgboost/collective/result.h"     // for Result

namespace xgboost::federated {
class FederatedService final : public Federated::Service {
 public:
  explicit FederatedService(std::int32_t world_size)
      : handler_{static_cast<std::size_t>(world_size)} {}

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

void RunServer(int port, std::size_t world_size, char const* server_key_file,
               char const* server_cert_file, char const* client_cert_file);

void RunInsecureServer(int port, std::size_t world_size);
}  // namespace xgboost::federated
