/*!
 * Copyright 2022 XGBoost contributors
 */
#pragma once

#include <federated.grpc.pb.h>

#include <condition_variable>
#include <mutex>

namespace xgboost {
namespace federated {

class FederatedService final : public Federated::Service {
 public:
  explicit FederatedService(int const world_size) : world_size_{world_size} {}

  grpc::Status Allgather(grpc::ServerContext* context, AllgatherRequest const* request,
                         AllgatherReply* reply) override;

  grpc::Status Allreduce(grpc::ServerContext* context, AllreduceRequest const* request,
                         AllreduceReply* reply) override;

  grpc::Status Broadcast(grpc::ServerContext* context, BroadcastRequest const* request,
                         BroadcastReply* reply) override;

 private:
  template <class Request, class Reply, class RequestFunctor>
  grpc::Status Handle(Request const* request, Reply* reply, RequestFunctor const& functor);

  int const world_size_;
  int received_{};
  int sent_{};
  std::string buffer_{};
  uint64_t sequence_number_{};
  mutable std::mutex mutex_;
  mutable std::condition_variable cv_;
};

void RunServer(int port, int world_size, char const* server_key_file, char const* server_cert_file,
               char const* client_cert_file);

void RunInsecureServer(int port, int world_size);

}  // namespace federated
}  // namespace xgboost
