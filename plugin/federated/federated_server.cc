/*!
 * Copyright 2022 XGBoost contributors
 */
#include "federated_server.h"

#include <grpcpp/grpcpp.h>
#include <grpcpp/server_builder.h>
#include <xgboost/logging.h>

#include <fstream>
#include <sstream>

#include "../../src/common/io.h"

namespace xgboost {
namespace federated {

class AllgatherFunctor {
 public:
  std::string const name{"Allgather"};

  explicit AllgatherFunctor(int const world_size) : world_size_{world_size} {}

  void operator()(AllgatherRequest const* request, std::string& buffer) const {
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

class AllreduceFunctor {
 public:
  std::string const name{"Allreduce"};

  void operator()(AllreduceRequest const* request, std::string& buffer) const {
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
      case DataType::INT8:
        Accumulate(reinterpret_cast<std::int8_t*>(&buffer[0]),
                   reinterpret_cast<std::int8_t const*>(input.data()), buffer.size(),
                   reduce_operation);
        break;
      case DataType::UINT8:
        Accumulate(reinterpret_cast<std::uint8_t*>(&buffer[0]),
                   reinterpret_cast<std::uint8_t const*>(input.data()), buffer.size(),
                   reduce_operation);
        break;
      case DataType::INT32:
        Accumulate(reinterpret_cast<std::int32_t*>(&buffer[0]),
                   reinterpret_cast<std::int32_t const*>(input.data()),
                   buffer.size() / sizeof(std::uint32_t), reduce_operation);
        break;
      case DataType::UINT32:
        Accumulate(reinterpret_cast<std::uint32_t*>(&buffer[0]),
                   reinterpret_cast<std::uint32_t const*>(input.data()),
                   buffer.size() / sizeof(std::uint32_t), reduce_operation);
        break;
      case DataType::INT64:
        Accumulate(reinterpret_cast<std::int64_t*>(&buffer[0]),
                   reinterpret_cast<std::int64_t const*>(input.data()),
                   buffer.size() / sizeof(std::int64_t), reduce_operation);
        break;
      case DataType::UINT64:
        Accumulate(reinterpret_cast<std::uint64_t*>(&buffer[0]),
                   reinterpret_cast<std::uint64_t const*>(input.data()),
                   buffer.size() / sizeof(std::uint64_t), reduce_operation);
        break;
      case DataType::FLOAT:
        Accumulate(reinterpret_cast<float*>(&buffer[0]),
                   reinterpret_cast<float const*>(input.data()), buffer.size() / sizeof(float),
                   reduce_operation);
        break;
      case DataType::DOUBLE:
        Accumulate(reinterpret_cast<double*>(&buffer[0]),
                   reinterpret_cast<double const*>(input.data()), buffer.size() / sizeof(double),
                   reduce_operation);
        break;
      default:
        throw std::invalid_argument("Invalid data type");
    }
  }
};

class BroadcastFunctor {
 public:
  std::string const name{"Broadcast"};

  void operator()(BroadcastRequest const* request, std::string& buffer) const {
    if (request->rank() == request->root()) {
      // Copy the send_buffer if this is the root.
      buffer = request->send_buffer();
    }
  }
};

grpc::Status FederatedService::Allgather(grpc::ServerContext* context,
                                         AllgatherRequest const* request, AllgatherReply* reply) {
  return Handle(request, reply, AllgatherFunctor{world_size_});
}

grpc::Status FederatedService::Allreduce(grpc::ServerContext* context,
                                         AllreduceRequest const* request, AllreduceReply* reply) {
  return Handle(request, reply, AllreduceFunctor{});
}

grpc::Status FederatedService::Broadcast(grpc::ServerContext* context,
                                         BroadcastRequest const* request, BroadcastReply* reply) {
  return Handle(request, reply, BroadcastFunctor{});
}

template <class Request, class Reply, class RequestFunctor>
grpc::Status FederatedService::Handle(Request const* request, Reply* reply,
                                      RequestFunctor const& functor) {
  // Pass through if there is only 1 client.
  if (world_size_ == 1) {
    reply->set_receive_buffer(request->send_buffer());
    return grpc::Status::OK;
  }

  std::unique_lock<std::mutex> lock(mutex_);

  auto const sequence_number = request->sequence_number();
  auto const rank = request->rank();

  LOG(INFO) << functor.name << " rank " << rank << ": waiting for current sequence number";
  cv_.wait(lock, [this, sequence_number] { return sequence_number_ == sequence_number; });

  LOG(INFO) << functor.name << " rank " << rank << ": handling request";
  functor(request, buffer_);
  received_++;

  if (received_ == world_size_) {
    LOG(INFO) << functor.name << " rank " << rank << ": all requests received";
    reply->set_receive_buffer(buffer_);
    sent_++;
    lock.unlock();
    cv_.notify_all();
    return grpc::Status::OK;
  }

  LOG(INFO) << functor.name << " rank " << rank << ": waiting for all clients";
  cv_.wait(lock, [this] { return received_ == world_size_; });

  LOG(INFO) << functor.name << " rank " << rank << ": sending reply";
  reply->set_receive_buffer(buffer_);
  sent_++;

  if (sent_ == world_size_) {
    LOG(INFO) << functor.name << " rank " << rank << ": all replies sent";
    sent_ = 0;
    received_ = 0;
    buffer_.clear();
    sequence_number_++;
    lock.unlock();
    cv_.notify_all();
  }

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
