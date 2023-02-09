/*!
 * Copyright 2022 XGBoost contributors
 */
#pragma once
#include <xgboost/json.h>

#include <string>

#include "../c_api/c_api_utils.h"
#include "in_memory_handler.h"

namespace xgboost {
namespace collective {

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

  bool IsDistributed() const override { return true; }
  bool IsFederated() const override { return false; }

  void AllGather(void* in_out, std::size_t size) override {
    std::string output;
    handler_.Allgather(static_cast<const char*>(in_out), size, &output, sequence_number_++,
                       GetRank());
    output.copy(static_cast<char*>(in_out), size);
  }

  void AllReduce(void* in_out, std::size_t size, DataType data_type, Operation operation) override {
    auto const bytes = size * GetTypeSize(data_type);
    std::string output;
    handler_.Allreduce(static_cast<const char*>(in_out), bytes, &output, sequence_number_++,
                       GetRank(), data_type, operation);
    output.copy(static_cast<char*>(in_out), bytes);
  }

  void Broadcast(void* in_out, std::size_t size, int root) override {
    std::string output;
    handler_.Broadcast(static_cast<const char*>(in_out), size, &output, sequence_number_++,
                       GetRank(), root);
    output.copy(static_cast<char*>(in_out), size);
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
