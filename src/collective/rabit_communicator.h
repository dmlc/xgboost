/**
 * Copyright 2022-2023 by XGBoost contributors
 */
#pragma once
#include <rabit/rabit.h>

#include <string>
#include <vector>

#include "communicator.h"
#include "xgboost/json.h"

namespace xgboost {
namespace collective {

class RabitCommunicator : public Communicator {
 public:
  static Communicator *Create(Json const &config) {
    std::vector<std::string> args_str;
    for (auto &items : get<Object const>(config)) {
      switch (items.second.GetValue().Type()) {
        case xgboost::Value::ValueKind::kString: {
          args_str.push_back(items.first + "=" + get<String const>(items.second));
          break;
        }
        case xgboost::Value::ValueKind::kInteger: {
          args_str.push_back(items.first + "=" + std::to_string(get<Integer const>(items.second)));
          break;
        }
        case xgboost::Value::ValueKind::kBoolean: {
          if (get<Boolean const>(items.second)) {
            args_str.push_back(items.first + "=1");
          } else {
            args_str.push_back(items.first + "=0");
          }
          break;
        }
        default:
          break;
      }
    }
    std::vector<char *> args;
    for (auto &key_value : args_str) {
      args.push_back(&key_value[0]);
    }
    if (!rabit::Init(static_cast<int>(args.size()), &args[0])) {
      LOG(FATAL) << "Failed to initialize Rabit";
    }
    return new RabitCommunicator(rabit::GetWorldSize(), rabit::GetRank());
  }

  RabitCommunicator(int world_size, int rank) : Communicator(world_size, rank) {}

  bool IsDistributed() const override { return rabit::IsDistributed(); }

  bool IsFederated() const override { return false; }

  void AllGather(void *send_receive_buffer, std::size_t size) override {
    auto const per_rank = size / GetWorldSize();
    auto const index = per_rank * GetRank();
    rabit::Allgather(static_cast<char *>(send_receive_buffer), size, index, per_rank, per_rank);
  }

  void AllReduce(void *send_receive_buffer, std::size_t count, DataType data_type,
                 Operation op) override {
    switch (data_type) {
      case DataType::kInt8:
        DoAllReduce<char>(send_receive_buffer, count, op);
        break;
      case DataType::kUInt8:
        DoAllReduce<unsigned char>(send_receive_buffer, count, op);
        break;
      case DataType::kInt32:
        DoAllReduce<std::int32_t>(send_receive_buffer, count, op);
        break;
      case DataType::kUInt32:
        DoAllReduce<std::uint32_t>(send_receive_buffer, count, op);
        break;
      case DataType::kInt64:
        DoAllReduce<std::int64_t>(send_receive_buffer, count, op);
        break;
      case DataType::kUInt64:
        DoAllReduce<std::uint64_t>(send_receive_buffer, count, op);
        break;
      case DataType::kFloat:
        DoAllReduce<float>(send_receive_buffer, count, op);
        break;
      case DataType::kDouble:
        DoAllReduce<double>(send_receive_buffer, count, op);
        break;
      default:
        LOG(FATAL) << "Unknown data type";
    }
  }

  void Broadcast(void *send_receive_buffer, std::size_t size, int root) override {
    rabit::Broadcast(send_receive_buffer, size, root);
  }

  std::string GetProcessorName() override { return rabit::GetProcessorName(); }

  void Print(const std::string &message) override { rabit::TrackerPrint(message); }

 protected:
  void Shutdown() override { rabit::Finalize(); }

 private:
  template <typename DType, std::enable_if_t<std::is_integral<DType>::value> * = nullptr>
  void DoBitwiseAllReduce(void *send_receive_buffer, std::size_t count, Operation op) {
    switch (op) {
      case Operation::kBitwiseAND:
        rabit::Allreduce<rabit::op::BitAND, DType>(static_cast<DType *>(send_receive_buffer),
                                                   count);
        break;
      case Operation::kBitwiseOR:
        rabit::Allreduce<rabit::op::BitOR, DType>(static_cast<DType *>(send_receive_buffer), count);
        break;
      case Operation::kBitwiseXOR:
        rabit::Allreduce<rabit::op::BitXOR, DType>(static_cast<DType *>(send_receive_buffer),
                                                   count);
        break;
      default:
        LOG(FATAL) << "Unknown allreduce operation";
    }
  }

  template <typename DType, std::enable_if_t<std::is_floating_point<DType>::value> * = nullptr>
  void DoBitwiseAllReduce(void *, std::size_t, Operation) {
    LOG(FATAL) << "Floating point types do not support bitwise operations.";
  }

  template <typename DType>
  void DoAllReduce(void *send_receive_buffer, std::size_t count, Operation op) {
    switch (op) {
      case Operation::kMax:
        rabit::Allreduce<rabit::op::Max, DType>(static_cast<DType *>(send_receive_buffer), count);
        break;
      case Operation::kMin:
        rabit::Allreduce<rabit::op::Min, DType>(static_cast<DType *>(send_receive_buffer), count);
        break;
      case Operation::kSum:
        rabit::Allreduce<rabit::op::Sum, DType>(static_cast<DType *>(send_receive_buffer), count);
        break;
      case Operation::kBitwiseAND:
      case Operation::kBitwiseOR:
      case Operation::kBitwiseXOR:
        DoBitwiseAllReduce<DType>(send_receive_buffer, count, op);
        break;
      default:
        LOG(FATAL) << "Unknown allreduce operation";
    }
  }
};
}  // namespace collective
}  // namespace xgboost
