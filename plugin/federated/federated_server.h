/*!
 * Copyright 2022 XGBoost contributors
 */
#pragma once

#include <federated.grpc.pb.h>

#include "../../src/collective/in_memory_handler.h"

namespace xgboost {
namespace federated {

class FederatedService final : public Federated::Service {
 public:
  explicit FederatedService(int const world_size) : handler_{world_size} {}

  grpc::Status Allgather(grpc::ServerContext* context, AllgatherRequest const* request,
                         AllgatherReply* reply) override;

  grpc::Status Allreduce(grpc::ServerContext* context, AllreduceRequest const* request,
                         AllreduceReply* reply) override;

  grpc::Status Broadcast(grpc::ServerContext* context, BroadcastRequest const* request,
                         BroadcastReply* reply) override;

 private:
  xgboost::collective::InMemoryHandler handler_;
};

void RunServer(int port, int world_size, char const* server_key_file, char const* server_cert_file,
               char const* client_cert_file);

void RunInsecureServer(int port, int world_size);

}  // namespace federated
}  // namespace xgboost
