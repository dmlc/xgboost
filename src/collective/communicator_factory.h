/*!
 * Copyright 2022 XGBoost contributors
 */
#pragma once
#include "communicator.h"

#if defined(XGBOOST_USE_FEDERATED)
#include "../../plugin/federated/federated_communicator.h"
#endif

namespace xgboost {
namespace collective {

enum class CommunicatorType { kUnknown, kRabit, kMPI, kFederated };

class CommunicatorFactory {
 public:
  static constexpr const char* kCommunicatorKey = "XGBOOST_COMMUNICATOR";

  static void Init(int argc, char* argv[]) {
    if (communicator_) {
      LOG(FATAL) << "Communicator can only be initialized once.";
    }

    auto type = GetTypeFromEnv();
    auto const arg = GetTypeFromArgs(argc, argv);
    if (arg != CommunicatorType::kUnknown) {
      type = arg;
    }
    switch (type) {
      case CommunicatorType::kRabit:
        LOG(FATAL) << "Not implemented yet.";
        break;
      case CommunicatorType::kMPI:
        LOG(FATAL) << "Not implemented yet.";
        break;
      case CommunicatorType::kFederated: {
#if defined(XGBOOST_USE_FEDERATED)
        FederatedCommunicatorFactory factory{argc, argv};
        communicator_.reset(factory.Create());
#else
        LOG(FATAL) << "XGBoost is not compiled with Federated Learning support.";
#endif
        break;
      }
      case CommunicatorType::kUnknown:
        LOG(FATAL) << "Unknown communicator type.";
        break;
    }
  }

  static void Finalize() { communicator_.reset(); }

  static Communicator* GetCommunicator() { return communicator_.get(); }

  static CommunicatorType GetTypeFromEnv() {
    auto* env = std::getenv(kCommunicatorKey);
    if (env != nullptr) {
      return StringToType(env);
    } else {
      return CommunicatorType::kUnknown;
    }
  }

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

  static thread_local std::unique_ptr<Communicator> communicator_;
};

}  // namespace collective
}  // namespace xgboost
