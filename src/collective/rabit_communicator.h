/*!
 * Copyright 2022 XGBoost contributors
 */
#pragma once
#include <rabit/rabit.h>

#include <string>

#include "communicator.h"

namespace xgboost {
namespace collective {

class RabitCommunicator : public Communicator {
 public:
  RabitCommunicator(int world_size, int rank) : Communicator(world_size, rank) {}

  ~RabitCommunicator() override { rabit::Finalize(); }

  bool IsDistributed() const override { return rabit::IsDistributed(); }

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

 private:
  template <typename DType>
  void DoAllReduce(void *send_receive_buffer, std::size_t count, Operation op) {
    switch (op) {
      case Operation::kMax:
        rabit::Allreduce<rabit::op::Max, DType>(static_cast<DType *>(send_receive_buffer), count);
        break;
      case Operation::kSum:
        rabit::Allreduce<rabit::op::Sum, DType>(static_cast<DType *>(send_receive_buffer), count);
        break;
      default:
        LOG(FATAL) << "Unknown allreduce operation";
    }
  }
};

class RabitCommunicatorFactory {
 public:
  RabitCommunicatorFactory(int argc, char *argv[]) {
    rabit::Init(argc, argv);
    world_size_ = rabit::GetWorldSize();
    rank_ = rabit::GetRank();
  }

  Communicator *Create() const { return new RabitCommunicator(world_size_, rank_); }

 private:
  int world_size_;
  int rank_;
};

}  // namespace collective
}  // namespace xgboost
