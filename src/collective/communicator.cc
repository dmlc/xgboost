/*!
 * Copyright 2022 XGBoost contributors
 */
#include <map>
#include "communicator.h"

#include "comm.h"
#include "in_memory_communicator.h"
#include "noop_communicator.h"
#include "rabit_communicator.h"

#if defined(XGBOOST_USE_FEDERATED)
  #include "../../plugin/federated/federated_communicator.h"
#endif

#include "../processing/processor.h"
processing::Processor *processor_instance;

namespace xgboost::collective {
thread_local std::unique_ptr<Communicator> Communicator::communicator_{new NoOpCommunicator()};
thread_local CommunicatorType Communicator::type_{};
thread_local std::string Communicator::nccl_path_{};

std::map<std::string, std::string> json_to_map(xgboost::Json const& config, std::string key) {
  auto json_map = xgboost::OptionalArg<xgboost::Object>(config, key, xgboost::JsonObject::Map{});
  std::map<std::string, std::string> params{};
  for (auto entry : json_map) {
    std::string text;
    xgboost::Value* value = &(entry.second.GetValue());
    if (value->Type() == xgboost::Value::ValueKind::kString) {
      text = reinterpret_cast<xgboost::String *>(value)->GetString();
    } else if (value->Type() == xgboost::Value::ValueKind::kInteger) {
      auto num = reinterpret_cast<xgboost::Integer *>(value)->GetInteger();
      text = std::to_string(num);
    } else if (value->Type() == xgboost::Value::ValueKind::kNumber) {
      auto num = reinterpret_cast<xgboost::Number *>(value)->GetNumber();
      text = std::to_string(num);
    } else if (value->Type() == xgboost::Value::ValueKind::kBoolean) {
        text = reinterpret_cast<xgboost::Boolean *>(value)->GetBoolean() ? "true" : "false";
    } else {
      text = "Unsupported type";
    }
    params[entry.first] = text;
  }
  return params;
}

void Communicator::Init(Json const& config) {
  auto nccl = OptionalArg<String>(config, "dmlc_nccl_path", std::string{DefaultNcclName()});
  nccl_path_ = nccl;

  auto type = GetTypeFromEnv();
  auto const arg = GetTypeFromConfig(config);
  if (arg != CommunicatorType::kUnknown) {
    type = arg;
  }
  if (type == CommunicatorType::kUnknown) {
    // Default to Rabit if unspecified.
    type = CommunicatorType::kRabit;
  }
  type_ = type;
  switch (type) {
    case CommunicatorType::kRabit: {
      communicator_.reset(RabitCommunicator::Create(config));
      break;
    }
    case CommunicatorType::kFederated: {
#if defined(XGBOOST_USE_FEDERATED)
  communicator_.reset(FederatedCommunicator::Create(config));
  // Get processor configs
  std::string plugin_name{};
  std::string loader_params_key{};
  std::string loader_params_map{};
  std::string proc_params_key{};
  std::string proc_params_map{};
  plugin_name = OptionalArg<String>(config, "plugin_name", plugin_name);
  // Initialize processor if plugin_name is provided
  if (!plugin_name.empty()) {
    std::map<std::string, std::string> loader_params = json_to_map(config, "loader_params");
    std::map<std::string, std::string> proc_params = json_to_map(config, "proc_params");
    processing::ProcessorLoader loader(loader_params);
    processor_instance = loader.load(plugin_name);
    processor_instance->Initialize(collective::GetRank() == 0, proc_params);
  }
#else
  LOG(FATAL) << "XGBoost is not compiled with Federated Learning support.";
#endif
    break;
  }

  case CommunicatorType::kInMemory:
  case CommunicatorType::kInMemoryNccl: {
    communicator_.reset(InMemoryCommunicator::Create(config));
    break;
  }
  case CommunicatorType::kUnknown:
    LOG(FATAL) << "Unknown communicator type.";
  }
}

#ifndef XGBOOST_USE_CUDA
void Communicator::Finalize() {
  communicator_->Shutdown();
  communicator_.reset(new NoOpCommunicator());
  if (processor_instance != nullptr) {
    processor_instance->Shutdown();
    processor_instance = nullptr;
  }
}
#endif
}  // namespace xgboost::collective
