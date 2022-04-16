#include <federated.grpc.pb.h>
#include <federated.pb.h>
#include <grpcpp/server_builder.h>

#include <condition_variable>
#include <mutex>
#include <utility>

namespace xgboost::federated {

template <class Request, class Reply>
class Operation {
 public:
  Operation(std::string name, int const world_size)
      : name_{std::move(name)}, world_size_{world_size} {}

  grpc::Status Operate(Request const* request, Reply* reply) {
    // Pass through if there is only 1 client.
    if (world_size_ == 1) {
      reply->set_receive_buffer(request->send_buffer());
      return grpc::Status::OK;
    }

    std::unique_lock lock(mutex_);

    auto const rank = request->rank();

    std::cout << name_ << " rank " << rank << ": on request\n";
    OnRequest(request);
    received_++;

    if (received_ == world_size_) {
      std::cout << name_ << " rank " << rank << ": all requests received\n";
      reply->set_receive_buffer(buffer_);
      sent_++;
      lock.unlock();
      cv_.notify_all();
      return grpc::Status::OK;
    }

    std::cout << name_ << " rank " << rank << ": waiting for all clients\n";
    cv_.wait(lock, [this] { return received_ == world_size_; });

    std::cout << name_ << " rank " << rank << ": sending reply\n";
    reply->set_receive_buffer(buffer_);
    sent_++;

    if (sent_ == world_size_) {
      std::cout << name_ << " rank " << rank << ": all replies sent\n";
      sent_ = 0;
      received_ = 0;
      buffer_.clear();
    }

    return grpc::Status::OK;
  }

 protected:
  virtual void OnRequest(Request const* request) = 0;

  std::string const name_;
  int const world_size_;
  int received_{};
  int sent_{};
  std::string buffer_{};
  mutable std::mutex mutex_;
  mutable std::condition_variable cv_;
};

class AllgatherOp : public Operation<AllgatherRequest, AllgatherReply> {
 public:
  explicit AllgatherOp(int const world_size)
      : Operation<AllgatherRequest, AllgatherReply>("Allgather", world_size) {}

 protected:
  void OnRequest(AllgatherRequest const* request) override {
    auto const rank = request->rank();
    auto const& send_buffer = request->send_buffer();
    auto const buffer_size = send_buffer.size();
    buffer_.resize(buffer_size * world_size_);
    buffer_.replace(rank * buffer_size, buffer_size, send_buffer);
  }
};

class AllreduceOp : public Operation<AllreduceRequest, AllreduceReply> {
 public:
  explicit AllreduceOp(int const world_size)
      : Operation<AllreduceRequest, AllreduceReply>("Allreduce", world_size) {}

 protected:
  void OnRequest(AllreduceRequest const* request) override {
    if (buffer_.empty()) {
      buffer_ = request->send_buffer();
    } else {
      Accumulate(request->send_buffer(), request->data_type(), request->reduce_operation());
    }
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
};

class BroadcastOp : public Operation<BroadcastRequest, BroadcastReply> {
 public:
  explicit BroadcastOp(int const world_size)
      : Operation<BroadcastRequest, BroadcastReply>("Broadcast", world_size) {}

 protected:
  void OnRequest(BroadcastRequest const* request) override {
    if (request->rank() == request->root()) {
      buffer_ = request->send_buffer();
    }
  }
};

class FederatedService final : public Federated::Service {
 public:
  explicit FederatedService(int const world_size)
      : allgather_op_{world_size}, allreduce_op_{world_size}, broadcast_op_{world_size} {}

  grpc::Status Allgather(grpc::ServerContext* context, AllgatherRequest const* request,
                         AllgatherReply* reply) override {
    return allgather_op_.Operate(request, reply);
  }

  grpc::Status Allreduce(grpc::ServerContext* context, AllreduceRequest const* request,
                         AllreduceReply* reply) override {
    return allreduce_op_.Operate(request, reply);
  }

  grpc::Status Broadcast(grpc::ServerContext* context, BroadcastRequest const* request,
                         BroadcastReply* reply) override {
    return broadcast_op_.Operate(request, reply);
  }

 private:
  AllgatherOp allgather_op_;
  AllreduceOp allreduce_op_;
  BroadcastOp broadcast_op_;
};

void RunServer(int port, int world_size) {
  std::string const server_address = "0.0.0.0:" + std::to_string(port);
  FederatedService service{world_size};

  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cout << "Federated server listening on " << server_address << ", world size " << world_size
            << '\n';

  server->Wait();
}
}  // namespace xgboost::federated

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: federated_server port world_size" << '\n';
    return 1;
  }
  auto port = std::stoi(argv[1]);
  auto world_size = std::stoi(argv[2]);
  xgboost::federated::RunServer(port, world_size);
  return 0;
}
