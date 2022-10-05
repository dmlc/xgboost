/*!
 * Copyright 2022 XGBoost contributors
 */
#pragma once
#include <string>

#include "communicator.h"

namespace xgboost {
namespace collective {

/**
 * A no-op communicator, used for non-distributed training.
 */
class NoOpCommunicator : public Communicator {
 public:
  NoOpCommunicator() : Communicator(1, 0) {}
  bool IsDistributed() const override { return false; }
  void AllReduce(void *send_receive_buffer, std::size_t count, DataType data_type,
                 Operation op) override {}
  void Broadcast(void *send_receive_buffer, std::size_t size, int root) override {}
  std::string GetProcessorName() override { return ""; }
  void Print(const std::string &message) override { LOG(CONSOLE) << message; }

 protected:
  void Shutdown() override {}
};

}  // namespace collective
}  // namespace xgboost
