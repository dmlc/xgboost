/**
 * Copyright 2022-2023, XGBoost contributors
 */
#include "in_memory_handler.h"

#include <algorithm>
#include <functional>
#include "comm.h"

namespace xgboost::collective {
/**
 * @brief Functor for allgather.
 */
class AllgatherFunctor {
 public:
  std::string const name{"Allgather"};

  AllgatherFunctor(std::int32_t world_size, std::int32_t rank)
      : world_size_{world_size}, rank_{rank} {}

  void operator()(char const* input, std::size_t bytes, std::string* buffer) const {
    if (buffer->empty()) {
      // Resize the buffer if this is the first request.
      buffer->resize(bytes * world_size_);
    }

    // Splice the input into the common buffer.
    buffer->replace(rank_ * bytes, bytes, input, bytes);
  }

 private:
  std::int32_t world_size_;
  std::int32_t rank_;
};

/**
 * @brief Functor for variable-length allgather.
 */
class AllgatherVFunctor {
 public:
  std::string const name{"AllgatherV"};

  AllgatherVFunctor(std::int32_t world_size, std::int32_t rank,
                    std::map<std::size_t, std::string_view>* data)
      : world_size_{world_size}, rank_{rank}, data_{data} {}

  void operator()(char const* input, std::size_t bytes, std::string* buffer) const {
    data_->emplace(rank_, std::string_view{input, bytes});
    if (data_->size() == static_cast<std::size_t>(world_size_)) {
      for (auto const& kv : *data_) {
        buffer->append(kv.second);
      }
      data_->clear();
    }
  }

 private:
  std::int32_t world_size_;
  std::int32_t rank_;
  std::map<std::size_t, std::string_view>* data_;
};

/**
 * @brief Functor for allreduce.
 */
class AllreduceFunctor {
 public:
  std::string const name{"Allreduce"};

  AllreduceFunctor(ArrayInterfaceHandler::Type dataType, Op operation)
      : data_type_{dataType}, operation_{operation} {}

  void operator()(char const* input, std::size_t bytes, std::string* buffer) const {
    if (buffer->empty()) {
      // Copy the input if this is the first request.
      buffer->assign(input, bytes);
    } else {
      auto n_bytes_type = DispatchDType(data_type_, [](auto t) { return sizeof(t); });
      // Apply the reduce_operation to the input and the buffer.
      Accumulate(input, bytes / n_bytes_type, &buffer->front());
    }
  }

 private:
  template <class T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
  void AccumulateBitwise(T* buffer, T const* input, std::size_t size, Op reduce_operation) const {
    switch (reduce_operation) {
      case Op::kBitwiseAND:
        std::transform(buffer, buffer + size, input, buffer, std::bit_and<T>());
        break;
      case Op::kBitwiseOR:
        std::transform(buffer, buffer + size, input, buffer, std::bit_or<T>());
        break;
      case Op::kBitwiseXOR:
        std::transform(buffer, buffer + size, input, buffer, std::bit_xor<T>());
        break;
      default:
        throw std::invalid_argument("Invalid reduce operation");
    }
  }

  template <class T, std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
  void AccumulateBitwise(T*, T const*, std::size_t, Op) const {
    LOG(FATAL) << "Floating point types do not support bitwise operations.";
  }

  template <class T>
  void Accumulate(T* buffer, T const* input, std::size_t size, Op reduce_operation) const {
    switch (reduce_operation) {
      case Op::kMax:
        std::transform(buffer, buffer + size, input, buffer,
                       [](T a, T b) { return std::max(a, b); });
        break;
      case Op::kMin:
        std::transform(buffer, buffer + size, input, buffer,
                       [](T a, T b) { return std::min(a, b); });
        break;
      case Op::kSum:
        std::transform(buffer, buffer + size, input, buffer, std::plus<T>());
        break;
      case Op::kBitwiseAND:
      case Op::kBitwiseOR:
      case Op::kBitwiseXOR:
        AccumulateBitwise(buffer, input, size, reduce_operation);
        break;
      default:
        throw std::invalid_argument("Invalid reduce operation");
    }
  }

  void Accumulate(char const* input, std::size_t size, char* buffer) const {
    using Type = ArrayInterfaceHandler::Type;
    switch (data_type_) {
      case Type::kI1:
        Accumulate(reinterpret_cast<std::int8_t*>(buffer),
                   reinterpret_cast<std::int8_t const*>(input), size, operation_);
        break;
      case Type::kU1:
        Accumulate(reinterpret_cast<std::uint8_t*>(buffer),
                   reinterpret_cast<std::uint8_t const*>(input), size, operation_);
        break;
      case Type::kI4:
        Accumulate(reinterpret_cast<std::int32_t*>(buffer),
                   reinterpret_cast<std::int32_t const*>(input), size, operation_);
        break;
      case Type::kU4:
        Accumulate(reinterpret_cast<std::uint32_t*>(buffer),
                   reinterpret_cast<std::uint32_t const*>(input), size, operation_);
        break;
      case Type::kI8:
        Accumulate(reinterpret_cast<std::int64_t*>(buffer),
                   reinterpret_cast<std::int64_t const*>(input), size, operation_);
        break;
      case Type::kU8:
        Accumulate(reinterpret_cast<std::uint64_t*>(buffer),
                   reinterpret_cast<std::uint64_t const*>(input), size, operation_);
        break;
      case Type::kF4:
        Accumulate(reinterpret_cast<float*>(buffer), reinterpret_cast<float const*>(input), size,
                   operation_);
        break;
      case Type::kF8:
        Accumulate(reinterpret_cast<double*>(buffer), reinterpret_cast<double const*>(input), size,
                   operation_);
        break;
      default:
        throw std::invalid_argument("Invalid data type");
    }
  }

 private:
  ArrayInterfaceHandler::Type data_type_;
  Op operation_;
};

/**
 * @brief Functor for broadcast.
 */
class BroadcastFunctor {
 public:
  std::string const name{"Broadcast"};

  BroadcastFunctor(std::int32_t rank, std::int32_t root) : rank_{rank}, root_{root} {}

  void operator()(char const* input, std::size_t bytes, std::string* buffer) const {
    if (rank_ == root_) {
      // Copy the input if this is the root.
      buffer->assign(input, bytes);
    }
  }

 private:
  std::int32_t rank_;
  std::int32_t root_;
};

void InMemoryHandler::Init(std::int32_t world_size, std::int32_t) {
  CHECK(world_size_ < world_size) << "In memory handler already initialized.";

  std::unique_lock<std::mutex> lock(mutex_);
  world_size_++;
  cv_.wait(lock, [this, world_size] { return world_size_ == world_size; });
  lock.unlock();
  cv_.notify_all();
}

void InMemoryHandler::Shutdown(uint64_t sequence_number, std::int32_t) {
  CHECK(world_size_ > 0) << "In memory handler already shutdown.";

  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this, sequence_number] { return sequence_number_ == sequence_number; });
  received_++;
  cv_.wait(lock, [this] { return received_ == world_size_; });

  received_ = 0;
  world_size_ = 0;
  sequence_number_ = 0;
  lock.unlock();
  cv_.notify_all();
}

void InMemoryHandler::Allgather(char const* input, std::size_t bytes, std::string* output,
                                std::size_t sequence_number, std::int32_t rank) {
  Handle(input, bytes, output, sequence_number, rank, AllgatherFunctor{world_size_, rank});
}

void InMemoryHandler::AllgatherV(char const* input, std::size_t bytes, std::string* output,
                                 std::size_t sequence_number, std::int32_t rank) {
  Handle(input, bytes, output, sequence_number, rank, AllgatherVFunctor{world_size_, rank, &aux_});
}

void InMemoryHandler::Allreduce(char const* input, std::size_t bytes, std::string* output,
                                std::size_t sequence_number, std::int32_t rank,
                                ArrayInterfaceHandler::Type data_type, Op op) {
  Handle(input, bytes, output, sequence_number, rank, AllreduceFunctor{data_type, op});
}

void InMemoryHandler::Broadcast(char const* input, std::size_t bytes, std::string* output,
                                std::size_t sequence_number, std::int32_t rank, std::int32_t root) {
  Handle(input, bytes, output, sequence_number, rank, BroadcastFunctor{rank, root});
}

template <class HandlerFunctor>
void InMemoryHandler::Handle(char const* input, std::size_t bytes, std::string* output,
                             std::size_t sequence_number, std::int32_t rank,
                             HandlerFunctor const& functor) {
  // Pass through if there is only 1 client.
  if (world_size_ == 1) {
    if (input != output->data()) {
      output->assign(input, bytes);
    }
    return;
  }

  std::unique_lock<std::mutex> lock(mutex_);

  LOG(DEBUG) << functor.name << " rank " << rank << ": waiting for current sequence number";
  cv_.wait(lock, [this, sequence_number] { return sequence_number_ == sequence_number; });

  LOG(DEBUG) << functor.name << " rank " << rank << ": handling request";
  functor(input, bytes, &buffer_);
  received_++;

  if (received_ == world_size_) {
    LOG(DEBUG) << functor.name << " rank " << rank << ": all requests received";
    output->assign(buffer_);
    sent_++;
    lock.unlock();
    cv_.notify_all();
    return;
  }

  LOG(DEBUG) << functor.name << " rank " << rank << ": waiting for all clients";
  cv_.wait(lock, [this] { return received_ == world_size_; });

  LOG(DEBUG) << functor.name << " rank " << rank << ": sending reply";
  output->assign(buffer_);
  sent_++;

  if (sent_ == world_size_) {
    LOG(DEBUG) << functor.name << " rank " << rank << ": all replies sent";
    sent_ = 0;
    received_ = 0;
    buffer_.clear();
    sequence_number_++;
    lock.unlock();
    cv_.notify_all();
  }
}
}  // namespace xgboost::collective
