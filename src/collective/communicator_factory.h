/*!
 * Copyright 2022 XGBoost contributors
 */
#pragma once
#include <xgboost/json.h>

#include <memory>
#include <string>

#include "communicator.h"

namespace xgboost {
namespace collective {

enum class CommunicatorType { kUnknown, kRabit, kMPI, kFederated };

class DeviceCommunicator;

class CommunicatorFactory {
 public:
  static void Init(Json const& config);

  static void Finalize();

  static CommunicatorFactory* GetInstance() { return instance_.get(); }

  Communicator* GetCommunicator() { return communicator_.get(); }

#if defined(XGBOOST_USE_CUDA)
  DeviceCommunicator* GetDeviceCommunicator(int device_ordinal);
#endif

  /** @brief Get the communicator type from environment variables. Visible for testing. */
  static CommunicatorType GetTypeFromEnv() {
    auto* env = std::getenv("XGBOOST_COMMUNICATOR");
    if (env != nullptr) {
      return StringToType(env);
    } else {
      return CommunicatorType::kUnknown;
    }
  }

  /** @brief Get the communicator type from runtime configuration. Visible for testing. */
  static CommunicatorType GetTypeFromConfig(Json const& config) {
    auto const& j_upper = config["XGBOOST_COMMUNICATOR"];
    if (IsA<String const>(j_upper)) {
      return StringToType(get<String const>(j_upper).c_str());
    }
    auto const& j_lower = config["xgboost_communicator"];
    if (IsA<String const>(j_lower)) {
      return StringToType(get<String const>(j_lower).c_str());
    }
    return CommunicatorType::kUnknown;
  }

 private:
  CommunicatorFactory(CommunicatorType type, Communicator* communicator);

 private:
  static CommunicatorType StringToType(char const* str) {
    CommunicatorType result = CommunicatorType::kUnknown;
    if (!CompareStringsCaseInsensitive("rabit", str)) {
      result = CommunicatorType::kRabit;
    } else if (!CompareStringsCaseInsensitive("mpi", str)) {
      result = CommunicatorType::kMPI;
    } else if (!CompareStringsCaseInsensitive("federated", str)) {
      result = CommunicatorType::kFederated;
    } else {
      LOG(FATAL) << "Unknown communicator type " << str;
    }
    return result;
  }

  static thread_local std::unique_ptr<CommunicatorFactory> instance_;
  CommunicatorType type_;
  std::unique_ptr<Communicator> communicator_;
#if defined(XGBOOST_USE_CUDA)
  std::unique_ptr<DeviceCommunicator> device_communicator_;
#endif
};

}  // namespace collective
}  // namespace xgboost
