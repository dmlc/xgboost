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
  bool IsFederated() const override { return false; }
  void AllGather(void *, std::size_t) override {}
  void AllReduce(void *, std::size_t, DataType, Operation) override {}
  void Broadcast(void *, std::size_t, int) override {}
  std::string GetProcessorName() override { return ""; }
  void Print(const std::string &message) override { LOG(CONSOLE) << message; }

 protected:
  void Shutdown() override {}
};

}  // namespace collective
}  // namespace xgboost
