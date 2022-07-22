/*!
 * Copyright 2022 XGBoost contributors
 */
#pragma once
#include <strings.h>

#include "communicator.h"

namespace xgboost {
namespace collective {

enum class CommunicatorType { kUnknown, kRabit, kMPI, kFederated };

class DeviceCommunicator;

class CommunicatorFactory {
 public:
  static constexpr char const* kCommunicatorKey = "XGBOOST_COMMUNICATOR";

  static void Init(int argc, char* argv[]);

  static void Finalize();

  static CommunicatorFactory* GetInstance() { return instance_.get(); }

  Communicator* GetCommunicator() { return communicator_.get(); }

#if defined(XGBOOST_USE_CUDA)
  DeviceCommunicator* GetDeviceCommunicator(int device_ordinal);
#endif

  /** @brief Get the communicator type from environment variables. Visible for testing. */
  static CommunicatorType GetTypeFromEnv() {
    auto* env = std::getenv(kCommunicatorKey);
    if (env != nullptr) {
      return StringToType(env);
    } else {
      return CommunicatorType::kUnknown;
    }
  }

  /** @brief Get the communicator type from arguments. Visible for testing. */
  static CommunicatorType GetTypeFromArgs(int argc, char* argv[]) {
    for (int i = 0; i < argc; ++i) {
      std::string const key_value = argv[i];
      auto const delimiter = key_value.find('=');
      if (delimiter != std::string::npos) {
        auto const key = key_value.substr(0, delimiter);
        auto const value = key_value.substr(delimiter + 1);
        if (!strcasecmp(key.c_str(), kCommunicatorKey)) {
          return StringToType(value.c_str());
        }
      }
    }
    return CommunicatorType::kUnknown;
  }

 private:
  CommunicatorFactory(CommunicatorType type, Communicator* communicator);

 private:
  static CommunicatorType StringToType(char const* str) {
    CommunicatorType result = CommunicatorType::kUnknown;
    if (!strcasecmp("rabit", str)) {
      result = CommunicatorType::kRabit;
    } else if (!strcasecmp("mpi", str)) {
      result = CommunicatorType::kMPI;
    } else if (!strcasecmp("federated", str)) {
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
