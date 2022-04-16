#pragma once
#include <federated.grpc.pb.h>
#include <federated.pb.h>
#include <grpcpp/grpcpp.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace xgboost {
namespace federated {

class FederatedClient {
 public:
  explicit FederatedClient(std::string const &server_address, int rank)
      : stub_{Federated::NewStub(
            grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials()))},
        rank_{rank} {}

  std::string Allgather(std::string const &send_buffer) {
    AllgatherRequest request;
    request.set_rank(rank_);
    request.set_send_buffer(send_buffer);

    AllgatherReply reply;
    grpc::ClientContext context;
    grpc::Status status = stub_->Allgather(&context, request, &reply);

    if (status.ok()) {
      return reply.receive_buffer();
    } else {
      std::cout << status.error_code() << ": " << status.error_message() << '\n';
      throw std::runtime_error("Allgather RPC failed");
    }
  }

  std::string Allreduce(std::string const &send_buffer, DataType data_type,
                        ReduceOperation reduce_operation) {
    AllreduceRequest request;
    request.set_rank(rank_);
    request.set_send_buffer(send_buffer);
    request.set_data_type(data_type);
    request.set_reduce_operation(reduce_operation);

    AllreduceReply reply;
    grpc::ClientContext context;
    grpc::Status status = stub_->Allreduce(&context, request, &reply);

    if (status.ok()) {
      return reply.receive_buffer();
    } else {
      std::cout << status.error_code() << ": " << status.error_message() << '\n';
      throw std::runtime_error("Allreduce RPC failed");
    }
  }

  std::string Broadcast(std::string const &send_buffer, int root) {
    BroadcastRequest request;
    request.set_rank(rank_);
    request.set_send_buffer(send_buffer);
    request.set_root(root);

    BroadcastReply reply;
    grpc::ClientContext context;
    grpc::Status status = stub_->Broadcast(&context, request, &reply);

    if (status.ok()) {
      return reply.receive_buffer();
    } else {
      std::cout << status.error_code() << ": " << status.error_message() << '\n';
      throw std::runtime_error("Broadcast RPC failed");
    }
  }

 private:
  std::unique_ptr<Federated::Stub> const stub_;
  int const rank_;
};

}  // namespace federated
}  // namespace xgboost
