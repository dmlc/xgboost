/*!
 * Copyright 2022 XGBoost contributors
 */
#pragma once
#include <xgboost/json.h>

#include <condition_variable>
#include <string>

#include "../c_api/c_api_utils.h"
#include "communicator.h"

namespace xgboost {
namespace collective {

class AllreduceFunctor {
 public:
  std::string const name{"Allreduce"};

  AllreduceFunctor(DataType dataType, Operation operation)
      : data_type_(dataType), operation_(operation) {}

  void operator()(char const* buffer, std::size_t size, std::string& shared_buffer) const {
    if (shared_buffer.empty()) {
      // Copy the buffer if this is the first request.
      std::copy(buffer, buffer + size * GetTypeSize(data_type_), std::back_inserter(shared_buffer));
    } else {
      // Apply the reduce_operation to the buffer and the common buffer.
      Accumulate(buffer, shared_buffer);
    }
  }

 private:
  template <class T>
  void Accumulate(T* buffer, T const* input, std::size_t n, Operation reduce_operation) const {
    switch (reduce_operation) {
      case Operation::kMax:
        std::transform(buffer, buffer + n, input, buffer, [](T a, T b) { return std::max(a, b); });
        break;
      case Operation::kMin:
        std::transform(buffer, buffer + n, input, buffer, [](T a, T b) { return std::min(a, b); });
        break;
      case Operation::kSum:
        std::transform(buffer, buffer + n, input, buffer, std::plus<T>());
        break;
      default:
        throw std::invalid_argument("Invalid reduce operation");
    }
  }

  void Accumulate(char const* buffer, std::string& shared_buffer) const {
    switch (data_type_) {
      case DataType::kInt8:
        Accumulate(reinterpret_cast<std::int8_t*>(&shared_buffer[0]),
                   reinterpret_cast<std::int8_t const*>(buffer), shared_buffer.size(), operation_);
        break;
      case DataType::kUInt8:
        Accumulate(reinterpret_cast<std::uint8_t*>(&shared_buffer[0]),
                   reinterpret_cast<std::uint8_t const*>(buffer), shared_buffer.size(), operation_);
        break;
      case DataType::kInt32:
        Accumulate(reinterpret_cast<std::int32_t*>(&shared_buffer[0]),
                   reinterpret_cast<std::int32_t const*>(buffer),
                   shared_buffer.size() / sizeof(std::uint32_t), operation_);
        break;
      case DataType::kUInt32:
        Accumulate(reinterpret_cast<std::uint32_t*>(&shared_buffer[0]),
                   reinterpret_cast<std::uint32_t const*>(buffer),
                   shared_buffer.size() / sizeof(std::uint32_t), operation_);
        break;
      case DataType::kInt64:
        Accumulate(reinterpret_cast<std::int64_t*>(&shared_buffer[0]),
                   reinterpret_cast<std::int64_t const*>(buffer),
                   shared_buffer.size() / sizeof(std::int64_t), operation_);
        break;
      case DataType::kUInt64:
        Accumulate(reinterpret_cast<std::uint64_t*>(&shared_buffer[0]),
                   reinterpret_cast<std::uint64_t const*>(buffer),
                   shared_buffer.size() / sizeof(std::uint64_t), operation_);
        break;
      case DataType::kFloat:
        Accumulate(reinterpret_cast<float*>(&shared_buffer[0]),
                   reinterpret_cast<float const*>(buffer), shared_buffer.size() / sizeof(float),
                   operation_);
        break;
      case DataType::kDouble:
        Accumulate(reinterpret_cast<double*>(&shared_buffer[0]),
                   reinterpret_cast<double const*>(buffer), shared_buffer.size() / sizeof(double),
                   operation_);
        break;
      default:
        throw std::invalid_argument("Invalid data type");
    }
  }

 private:
  DataType data_type_;
  Operation operation_;
};

class BroadcastFunctor {
 public:
  std::string const name{"Broadcast"};

  BroadcastFunctor(int rank, int root) : rank_(rank), root_(root) {}

  void operator()(char const* buffer, std::size_t size, std::string& shared_buffer) const {
    if (rank_ == root_) {
      // Copy the buffer if this is the root.
      std::copy(buffer, buffer + size, std::back_inserter(shared_buffer));
    }
  }

 private:
  int rank_;
  int root_;
};

class InMemoryHandler {
 public:
  void Init(int world_size, int rank) {
    std::unique_lock<std::mutex> lock(mutex_);
    world_size_++;
    LOG(INFO) << "Rank " << rank << ": waiting for all ranks to initialize";
    cv_.wait(lock, [this, world_size] { return world_size_ == world_size; });
    lock.unlock();
    cv_.notify_all();
  }

  void Shutdown(uint64_t sequence_number, int rank) {
    std::unique_lock<std::mutex> lock(mutex_);

    LOG(INFO) << "Rank " << rank << ": waiting for current sequence number";
    cv_.wait(lock, [this, sequence_number] { return sequence_number_ == sequence_number; });

    LOG(INFO) << "Rank " << rank << ": handling shutdown request";
    received_++;

    LOG(INFO) << "Rank " << rank << ": waiting for all clients";
    cv_.wait(lock, [this] { return received_ == world_size_; });

    received_ = 0;
    world_size_ = 0;
    sequence_number_ = 0;
    lock.unlock();
    cv_.notify_all();
  }

  void Allreduce(void* buffer, std::size_t size, std::size_t sequence_number, int rank,
                 DataType data_type, Operation op) {
    return Handle(static_cast<char*>(buffer), size, sequence_number, rank,
                  AllreduceFunctor{data_type, op});
  }

  void Broadcast(void* buffer, std::size_t size, std::size_t sequence_number, int rank, int root) {
    return Handle(static_cast<char*>(buffer), size, sequence_number, rank,
                  BroadcastFunctor{rank, root});
  }

 private:
  template <class HandlerFunctor>
  void Handle(void* buffer, std::size_t size, std::size_t sequence_number, int rank,
              HandlerFunctor const& functor) {
    // Pass through if there is only 1 client.
    if (world_size_ == 1) {
      return;
    }

    std::unique_lock<std::mutex> lock(mutex_);

    LOG(INFO) << functor.name << " rank " << rank << ": waiting for current sequence number";
    cv_.wait(lock, [this, sequence_number] { return sequence_number_ == sequence_number; });

    LOG(INFO) << functor.name << " rank " << rank << ": handling request";
    functor(static_cast<char const*>(buffer), size, shared_buffer_);
    received_++;

    if (received_ == world_size_) {
      LOG(INFO) << functor.name << " rank " << rank << ": all requests received";
      shared_buffer_.copy(static_cast<char*>(buffer), shared_buffer_.size());
      sent_++;
      lock.unlock();
      cv_.notify_all();
      return;
    }

    LOG(INFO) << functor.name << " rank " << rank << ": waiting for all clients";
    cv_.wait(lock, [this] { return received_ == world_size_; });

    LOG(INFO) << functor.name << " rank " << rank << ": sending reply";
    shared_buffer_.copy(static_cast<char*>(buffer), shared_buffer_.size());
    sent_++;

    if (sent_ == world_size_) {
      LOG(INFO) << functor.name << " rank " << rank << ": all replies sent";
      sent_ = 0;
      received_ = 0;
      shared_buffer_.clear();
      sequence_number_++;
      lock.unlock();
      cv_.notify_all();
    }
  }

  int world_size_;
  int received_{};
  int sent_{};
  std::string shared_buffer_{};
  uint64_t sequence_number_{};
  mutable std::mutex mutex_;
  mutable std::condition_variable cv_;
};

/**
 * An in-memory communicator, useful for testing.
 */
class InMemoryCommunicator : public Communicator {
 public:
  /**
   * @brief Create a new communicator based on JSON configuration.
   * @param config JSON configuration.
   * @return Communicator as specified by the JSON configuration.
   */
  static Communicator* Create(Json const& config) {
    int world_size{0};
    int rank{-1};

    // Parse environment variables first.
    auto* value = getenv("IN_MEMORY_WORLD_SIZE");
    if (value != nullptr) {
      world_size = std::stoi(value);
    }
    value = getenv("IN_MEMORY_RANK");
    if (value != nullptr) {
      rank = std::stoi(value);
    }

    // Runtime configuration overrides, optional as users can specify them as env vars.
    world_size = static_cast<int>(OptionalArg<Integer>(config, "in_memory_world_size",
                                                       static_cast<Integer::Int>(world_size)));
    rank = static_cast<int>(
        OptionalArg<Integer>(config, "in_memory_rank", static_cast<Integer::Int>(rank)));

    if (world_size == 0) {
      LOG(FATAL) << "Federated world size must be set.";
    }
    if (rank == -1) {
      LOG(FATAL) << "Federated rank must be set.";
    }
    return new InMemoryCommunicator(world_size, rank);
  }

  InMemoryCommunicator(int world_size, int rank) : Communicator(world_size, rank) {
    handler_.Init(world_size, rank);
  }

  ~InMemoryCommunicator() override { handler_.Shutdown(sequence_number_++, GetRank()); }

  bool IsDistributed() const override { return false; }
  bool IsFederated() const override { return false; }

  void AllReduce(void* buffer, std::size_t size, DataType data_type, Operation operation) override {
    handler_.Allreduce(buffer, size, sequence_number_++, GetRank(), data_type, operation);
  }

  void Broadcast(void* buffer, std::size_t size, int root) override {
    handler_.Broadcast(buffer, size, sequence_number_++, GetRank(), root);
  }

  std::string GetProcessorName() override { return "rank" + std::to_string(GetRank()); }

  void Print(const std::string& message) override { LOG(CONSOLE) << message; }

 protected:
  void Shutdown() override {}

 private:
  static InMemoryHandler handler_;
  uint64_t sequence_number_{};
};

}  // namespace collective
}  // namespace xgboost
