#include <federation.grpc.pb.h>
#include <federation.pb.h>
#include <grpcpp/grpcpp.h>

#include <cstdio>
#include <cstdlib>
#include <string>

class FederationClient {
 public:
  explicit FederationClient(const std::shared_ptr<grpc::Channel> &channel)
      : stub_(xgboost::federated::Federation::NewStub(channel)) {}

  std::string Allreduce(const std::string &send_buffer) {
    xgboost::federated::AllreduceRequest request;
    request.set_send_buffer(send_buffer);
    request.set_data_type(xgboost::federated::INT);
    request.set_reduce_operation(xgboost::federated::SUM);

    xgboost::federated::AllreduceReply reply;
    grpc::ClientContext context;
    grpc::Status status = stub_->Allreduce(&context, request, &reply);

    if (status.ok()) {
      return reply.receive_buffer();
    } else {
      std::cout << status.error_code() << ": " << status.error_message() << std::endl;
      return "RPC failed";
    }
  }

 private:
  std::unique_ptr<xgboost::federated::Federation::Stub> stub_;
};

int main(int argc, char **argv) {
  FederationClient client(
      grpc::CreateChannel("localhost:50051", grpc::InsecureChannelCredentials()));

  for (int i = 1; i <= 10; i++) {
    int data[] = {1 * i, 2 * i, 3 * i, 4 * i, 5 * i};
    int n = sizeof(data) / sizeof(data[0]);
    std::string send_buffer(reinterpret_cast<char const *>(data), sizeof(data));
    std::string receive_buffer = client.Allreduce(send_buffer);
    int *result = reinterpret_cast<int *>(receive_buffer.data());
    std::copy(result, result + n, std::ostream_iterator<int>(std::cout, " "));
    std::cout << '\n';
  }

  return 0;
}
