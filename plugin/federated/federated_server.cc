/*!
 * Copyright 2022 XGBoost contributors
 */
#include <federated.grpc.pb.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/server_builder.h>

#include <condition_variable>
#include <fstream>
#include <mutex>
#include <sstream>

namespace xgboost::federated {

class AllgatherHandler {
 public:
  std::string const name{"Allgather"};

  explicit AllgatherHandler(int const world_size) : world_size_{world_size} {}

  void Handle(AllgatherRequest const* request, std::string& buffer) const {
    auto const rank = request->rank();
    auto const& send_buffer = request->send_buffer();
    auto const send_size = send_buffer.size();
    // Resize the buffer if this is the first request.
    if (buffer.size() != send_size * world_size_) {
      buffer.resize(send_size * world_size_);
    }
    // Splice the send_buffer into the common buffer.
    buffer.replace(rank * send_size, send_size, send_buffer);
  }

 private:
  int const world_size_;
};

class AllreduceHandler {
 public:
  std::string const name{"Allreduce"};

  void Handle(AllreduceRequest const* request, std::string& buffer) const {
    if (buffer.empty()) {
      // Copy the send_buffer if this is the first request.
      buffer = request->send_buffer();
    } else {
      // Apply the reduce_operation to the send_buffer and the common buffer.
      Accumulate(buffer, request->send_buffer(), request->data_type(), request->reduce_operation());
    }
  }

 private:
  template <class T>
  void Accumulate(T* buffer, T const* input, std::size_t n,
                  ReduceOperation reduce_operation) const {
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

  void Accumulate(std::string& buffer, std::string const& input, DataType data_type,
                  ReduceOperation reduce_operation) const {
    switch (data_type) {
      case DataType::CHAR:
        Accumulate(buffer.data(), input.data(), buffer.size(), reduce_operation);
        break;
      case DataType::UCHAR:
        Accumulate(reinterpret_cast<unsigned char*>(buffer.data()),
                   reinterpret_cast<unsigned char const*>(input.data()), buffer.size(),
                   reduce_operation);
        break;
      case DataType::INT:
        Accumulate(reinterpret_cast<int*>(buffer.data()),
                   reinterpret_cast<int const*>(input.data()), buffer.size() / sizeof(int),
                   reduce_operation);
        break;
      case DataType::UINT:
        Accumulate(reinterpret_cast<unsigned int*>(buffer.data()),
                   reinterpret_cast<unsigned int const*>(input.data()),
                   buffer.size() / sizeof(unsigned int), reduce_operation);
        break;
      case DataType::LONG:
        Accumulate(reinterpret_cast<long*>(buffer.data()),
                   reinterpret_cast<long const*>(input.data()), buffer.size() / sizeof(long),
                   reduce_operation);
        break;
      case DataType::ULONG:
        Accumulate(reinterpret_cast<unsigned long*>(buffer.data()),
                   reinterpret_cast<unsigned long const*>(input.data()),
                   buffer.size() / sizeof(unsigned long), reduce_operation);
        break;
      case DataType::FLOAT:
        Accumulate(reinterpret_cast<float*>(buffer.data()),
                   reinterpret_cast<float const*>(input.data()), buffer.size() / sizeof(float),
                   reduce_operation);
        break;
      case DataType::DOUBLE:
        Accumulate(reinterpret_cast<double*>(buffer.data()),
                   reinterpret_cast<double const*>(input.data()), buffer.size() / sizeof(double),
                   reduce_operation);
        break;
      case DataType::LONGLONG:
        Accumulate(reinterpret_cast<long long*>(buffer.data()),
                   reinterpret_cast<long long const*>(input.data()),
                   buffer.size() / sizeof(long long), reduce_operation);
        break;
      case DataType::ULONGLONG:
        Accumulate(reinterpret_cast<unsigned long long*>(buffer.data()),
                   reinterpret_cast<unsigned long long const*>(input.data()),
                   buffer.size() / sizeof(unsigned long long), reduce_operation);
        break;
      default:
        throw std::invalid_argument("Invalid data type");
    }
  }
};

class BroadcastHandler {
 public:
  std::string const name{"Broadcast"};

  static void Handle(BroadcastRequest const* request, std::string& buffer) {
    if (request->rank() == request->root()) {
      // Copy the send_buffer if this is the root.
      buffer = request->send_buffer();
    }
  }
};

class FederatedService final : public Federated::Service {
 public:
  explicit FederatedService(int const world_size)
      : world_size_{world_size},
        allgather_handler_{world_size},
        allreduce_handler_{},
        broadcast_handler_{} {}

  grpc::Status Allgather(grpc::ServerContext* context, AllgatherRequest const* request,
                         AllgatherReply* reply) override {
    return Handle(request, reply, allgather_handler_);
  }

  grpc::Status Allreduce(grpc::ServerContext* context, AllreduceRequest const* request,
                         AllreduceReply* reply) override {
    return Handle(request, reply, allreduce_handler_);
  }

  grpc::Status Broadcast(grpc::ServerContext* context, BroadcastRequest const* request,
                         BroadcastReply* reply) override {
    return Handle(request, reply, broadcast_handler_);
  }

 private:
  template <class Request, class Reply, class RequestHandler>
  grpc::Status Handle(Request const* request, Reply* reply, RequestHandler const& handler) {
    // Pass through if there is only 1 client.
    if (world_size_ == 1) {
      reply->set_receive_buffer(request->send_buffer());
      return grpc::Status::OK;
    }

    std::unique_lock lock(mutex_);

    auto const sequence_number = request->sequence_number();
    auto const rank = request->rank();

    std::cout << handler.name << " rank " << rank << ": waiting for current sequence number\n";
    cv_.wait(lock, [this, sequence_number] { return sequence_number_ == sequence_number; });

    std::cout << handler.name << " rank " << rank << ": handling request\n";
    handler.Handle(request, buffer_);
    received_++;

    if (received_ == world_size_) {
      std::cout << handler.name << " rank " << rank << ": all requests received\n";
      reply->set_receive_buffer(buffer_);
      sent_++;
      lock.unlock();
      cv_.notify_all();
      return grpc::Status::OK;
    }

    std::cout << handler.name << " rank " << rank << ": waiting for all clients\n";
    cv_.wait(lock, [this] { return received_ == world_size_; });

    std::cout << handler.name << " rank " << rank << ": sending reply\n";
    reply->set_receive_buffer(buffer_);
    sent_++;

    if (sent_ == world_size_) {
      std::cout << handler.name << " rank " << rank << ": all replies sent\n";
      sent_ = 0;
      received_ = 0;
      buffer_.clear();
      sequence_number_++;
      lock.unlock();
      cv_.notify_all();
    }

    return grpc::Status::OK;
  }

  int const world_size_;
  AllgatherHandler allgather_handler_;
  AllreduceHandler allreduce_handler_;
  BroadcastHandler broadcast_handler_;
  int received_{};
  int sent_{};
  std::string buffer_{};
  uint64_t sequence_number_{};
  mutable std::mutex mutex_;
  mutable std::condition_variable cv_;
};

std::string ReadFile(std::string const& path) {
  auto stream = std::ifstream(path.data());
  std::ostringstream out;
  out << stream.rdbuf();
  return out.str();
}

void RunServer(int port, int world_size, std::string const& ca_cert_file,
               std::string const& key_file, std::string const& cert_file) {
  std::string const server_address = "0.0.0.0:" + std::to_string(port);
  FederatedService service{world_size};

  grpc::ServerBuilder builder;
  auto options =
      grpc::SslServerCredentialsOptions(GRPC_SSL_REQUEST_AND_REQUIRE_CLIENT_CERTIFICATE_AND_VERIFY);
  options.pem_root_certs = ReadFile(ca_cert_file);
  auto key = grpc::SslServerCredentialsOptions::PemKeyCertPair();
  key.private_key = ReadFile(key_file);
  key.cert_chain = ReadFile(cert_file);
  options.pem_key_cert_pairs.push_back(key);
  builder.AddListeningPort(server_address, grpc::SslServerCredentials(options));
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cout << "Federated server listening on " << server_address << ", world size " << world_size
            << '\n';

  server->Wait();
}
}  // namespace xgboost::federated

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "Usage: federated_server port world_size ca_cert_file key_file cert_file" << '\n';
    return 1;
  }
  auto port = std::stoi(argv[1]);
  auto world_size = std::stoi(argv[2]);
  std::string ca_cert_file = argv[3];
  std::string key_file = argv[4];
  std::string cert_file = argv[5];
  xgboost::federated::RunServer(port, world_size, ca_cert_file, key_file, cert_file);
  return 0;
}
