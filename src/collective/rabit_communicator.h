/*!
 * Copyright 2022 XGBoost contributors
 */
#pragma once
#include <rabit/rabit.h>

#include "communicator.h"

namespace xgboost {
namespace collective {

class RabitCommunicator : public Communicator {
 public:
  RabitCommunicator(int world_size, int rank) : Communicator(world_size, rank) {}

  ~RabitCommunicator() override { rabit::Finalize(); }

  void AllReduce(void *send_receive_buffer, std::size_t count, DataType data_type,
                 Operation op) override {
    switch (data_type) {
      case DataType::kInt:
        DoAllReduce<int>(send_receive_buffer, count, op);
        break;
      case DataType::kFloat:
        DoAllReduce<float>(send_receive_buffer, count, op);
        break;
      case DataType::kDouble:
        DoAllReduce<double>(send_receive_buffer, count, op);
        break;
      case DataType::kSizeT:
        DoAllReduce<std::size_t>(send_receive_buffer, count, op);
        break;
      default:
        LOG(FATAL) << "Unknown data type";
    }
  }

  void Broadcast(void *send_receive_buffer, std::size_t size, int root) override {
    rabit::Broadcast(send_receive_buffer, size, root);
  }

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
