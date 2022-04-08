#include <federation.grpc.pb.h>
#include <federation.pb.h>
#include <grpcpp/server_builder.h>

#include <condition_variable>
#include <mutex>

namespace xgboost::federated {

class FederationService final : public Federation::Service {
 public:
  explicit FederationService(int const world_size) : world_size_(world_size) {}

  grpc::Status Allreduce(grpc::ServerContext* context, AllreduceRequest const* request,
                         AllreduceReply* reply) override {
    // Pass through if there is only 1 client.
    if (world_size_ == 1) {
      reply->set_receive_buffer(request->send_buffer());
      return grpc::Status::OK;
    }

    std::unique_lock lock(mutex_);

    // Wait for all previous replies have been sent.
    cv_.wait(lock, [this] { return sent_ == 0; });

    if (received_ == 0) {
      // Copy the send_buffer if this is the first client.
      buffer_ = request->send_buffer();
    } else {
      // Accumulate the send_buffer into the common buffer.
      Accumulate(request->send_buffer(), request->data_type(), request->reduce_operation());
    }
    received_++;
    // If all clients have been received, send the reply and notify all.
    if (received_ == world_size_) {
      received_ = 0;
      sent_++;
      reply->set_receive_buffer(buffer_);
      lock.unlock();
      cv_.notify_all();
      return grpc::Status::OK;
    }

    // Wait for all the clients to be received.
    cv_.wait(lock, [this] { return received_ == 0; });
    sent_++;
    reply->set_receive_buffer(buffer_);
    if (sent_ == world_size_) {
      sent_ = 0;
      lock.unlock();
      cv_.notify_all();
    }
    return grpc::Status::OK;
  }

 private:
  template <class T>
  void Accumulate(T* buffer, T const* input, std::size_t n, ReduceOperation reduce_operation) {
    switch (reduce_operation) {
      case ReduceOperation::MAX:
        std::transform(buffer, buffer + n, input, buffer, [](T a, T b) { return std::max(a, b); });
        break;
      case ReduceOperation::MIN:
        std::transform(buffer, buffer + n, input, buffer, [](T a, T b) { return std::min(a, b); });
        break;
      case ReduceOperation::SUM:
        std::transform(buffer, buffer + n, input, buffer, std::plus<T>());
        break;
      default:
        throw std::invalid_argument("Invalid reduce operation");
    }
  }

  void Accumulate(std::string const& input, DataType data_type, ReduceOperation reduce_operation) {
    switch (data_type) {
      case DataType::CHAR:
        Accumulate(buffer_.data(), input.data(), buffer_.size(), reduce_operation);
        break;
      case DataType::UCHAR:
        Accumulate(reinterpret_cast<unsigned char*>(buffer_.data()),
                   reinterpret_cast<unsigned char const*>(input.data()), buffer_.size(),
                   reduce_operation);
        break;
      case DataType::INT:
        Accumulate(reinterpret_cast<int*>(buffer_.data()),
                   reinterpret_cast<int const*>(input.data()), buffer_.size() / sizeof(int),
                   reduce_operation);
        break;
      case DataType::UINT:
        Accumulate(reinterpret_cast<unsigned int*>(buffer_.data()),
                   reinterpret_cast<unsigned int const*>(input.data()),
                   buffer_.size() / sizeof(unsigned int), reduce_operation);
        break;
      case DataType::LONG:
        Accumulate(reinterpret_cast<long*>(buffer_.data()),
                   reinterpret_cast<long const*>(input.data()), buffer_.size() / sizeof(long),
                   reduce_operation);
        break;
      case DataType::ULONG:
        Accumulate(reinterpret_cast<unsigned long*>(buffer_.data()),
                   reinterpret_cast<unsigned long const*>(input.data()),
                   buffer_.size() / sizeof(unsigned long), reduce_operation);
        break;
      case DataType::FLOAT:
        Accumulate(reinterpret_cast<float*>(buffer_.data()),
                   reinterpret_cast<float const*>(input.data()), buffer_.size() / sizeof(float),
                   reduce_operation);
        break;
      case DataType::DOUBLE:
        Accumulate(reinterpret_cast<double*>(buffer_.data()),
                   reinterpret_cast<double const*>(input.data()), buffer_.size() / sizeof(double),
                   reduce_operation);
        break;
      case DataType::LONGLONG:
        Accumulate(reinterpret_cast<long long*>(buffer_.data()),
                   reinterpret_cast<long long const*>(input.data()),
                   buffer_.size() / sizeof(long long), reduce_operation);
        break;
      case DataType::ULONGLONG:
        Accumulate(reinterpret_cast<unsigned long long*>(buffer_.data()),
                   reinterpret_cast<unsigned long long const*>(input.data()),
                   buffer_.size() / sizeof(unsigned long long), reduce_operation);
        break;
      default:
        throw std::invalid_argument("Invalid data type");
    }
  }

  int const world_size_;
  int received_{};
  int sent_{};
  std::string buffer_{};
  mutable std::mutex mutex_;
  mutable std::condition_variable cv_;
};

void RunServer(int world_size) {
  std::string const server_address{"0.0.0.0:50051"};
  FederationService service{world_size};

  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << " with world size " << world_size
            << '\n';

  server->Wait();
}
}  // namespace xgboost::federated

int main(int argc, char** argv) {
  auto world_size{1};
  if (argc > 1) {
    world_size = std::stoi(argv[1]);
  }
  xgboost::federated::RunServer(world_size);
  return 0;
}
