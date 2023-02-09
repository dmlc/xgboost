/*!
 * Copyright 2022 XGBoost contributors
 */
#include "federated_server.h"

#include <grpcpp/grpcpp.h>
#include <grpcpp/server_builder.h>
#include <xgboost/logging.h>

#include <sstream>

#include "../../src/common/io.h"

namespace xgboost {
namespace federated {

grpc::Status FederatedService::Allgather(grpc::ServerContext* context,
                                         AllgatherRequest const* request, AllgatherReply* reply) {
  handler_.Allgather(request->send_buffer().data(), request->send_buffer().size(),
                     reply->mutable_receive_buffer(), request->sequence_number(), request->rank());
  return grpc::Status::OK;
}

grpc::Status FederatedService::Allreduce(grpc::ServerContext* context,
                                         AllreduceRequest const* request, AllreduceReply* reply) {
  handler_.Allreduce(request->send_buffer().data(), request->send_buffer().size(),
                     reply->mutable_receive_buffer(), request->sequence_number(), request->rank(),
                     static_cast<xgboost::collective::DataType>(request->data_type()),
                     static_cast<xgboost::collective::Operation>(request->reduce_operation()));
  return grpc::Status::OK;
}

grpc::Status FederatedService::Broadcast(grpc::ServerContext* context,
                                         BroadcastRequest const* request, BroadcastReply* reply) {
  handler_.Broadcast(request->send_buffer().data(), request->send_buffer().size(),
                     reply->mutable_receive_buffer(), request->sequence_number(), request->rank(),
                     request->root());
  return grpc::Status::OK;
}

void RunServer(int port, int world_size, char const* server_key_file, char const* server_cert_file,
               char const* client_cert_file) {
  std::string const server_address = "0.0.0.0:" + std::to_string(port);
  FederatedService service{world_size};

  grpc::ServerBuilder builder;
  auto options =
      grpc::SslServerCredentialsOptions(GRPC_SSL_REQUEST_AND_REQUIRE_CLIENT_CERTIFICATE_AND_VERIFY);
  options.pem_root_certs = xgboost::common::ReadAll(client_cert_file);
  auto key = grpc::SslServerCredentialsOptions::PemKeyCertPair();
  key.private_key = xgboost::common::ReadAll(server_key_file);
  key.cert_chain = xgboost::common::ReadAll(server_cert_file);
  options.pem_key_cert_pairs.push_back(key);
  builder.SetMaxReceiveMessageSize(std::numeric_limits<int>::max());
  builder.AddListeningPort(server_address, grpc::SslServerCredentials(options));
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  LOG(CONSOLE) << "Federated server listening on " << server_address << ", world size "
               << world_size;

  server->Wait();
}

void RunInsecureServer(int port, int world_size) {
  std::string const server_address = "0.0.0.0:" + std::to_string(port);
  FederatedService service{world_size};

  grpc::ServerBuilder builder;
  builder.SetMaxReceiveMessageSize(std::numeric_limits<int>::max());
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  LOG(CONSOLE) << "Insecure federated server listening on " << server_address << ", world size "
               << world_size;

  server->Wait();
}

}  // namespace federated
}  // namespace xgboost
