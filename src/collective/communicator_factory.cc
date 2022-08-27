/*!
 * Copyright 2022 XGBoost contributors
 */
#include "communicator_factory.h"

#include "rabit_communicator.h"

#if defined(XGBOOST_USE_FEDERATED)
#include "../../plugin/federated/federated_communicator.h"
#endif

namespace xgboost {
namespace collective {

#ifndef XGBOOST_USE_CUDA
thread_local std::unique_ptr<CommunicatorFactory> CommunicatorFactory::instance_{};

CommunicatorFactory::CommunicatorFactory(CommunicatorType type, Communicator* communicator)
    : type_{type}, communicator_{communicator} {}

void CommunicatorFactory::Init(int argc, char* argv[]) {
  if (instance_) {
    LOG(FATAL) << "Communicator factory can only be initialized once.";
  }

  auto type = GetTypeFromEnv();
  auto const arg = GetTypeFromArgs(argc, argv);
  if (arg != CommunicatorType::kUnknown) {
    type = arg;
  }
  if (type == CommunicatorType::kUnknown) {
    // Default to Rabit if unspecified.
    type = CommunicatorType::kRabit;
  }
  switch (type) {
    case CommunicatorType::kRabit: {
      RabitCommunicatorFactory factory{argc, argv};
      auto* comm = factory.Create();
      instance_.reset(new CommunicatorFactory(type, comm));
      break;
    }
    case CommunicatorType::kMPI:
      LOG(FATAL) << "Not implemented yet.";
      break;
    case CommunicatorType::kFederated: {
#if defined(XGBOOST_USE_FEDERATED)
      FederatedCommunicatorFactory factory{argc, argv};
      auto* comm = factory.Create();
      instance_.reset(new CommunicatorFactory(type, comm));
#else
      LOG(FATAL) << "XGBoost is not compiled with Federated Learning support.";
#endif
      break;
    }
    case CommunicatorType::kUnknown:
      LOG(FATAL) << "Unknown communicator type.";
  }
}

void CommunicatorFactory::Finalize() { instance_.reset(); }
#endif

}  // namespace collective
}  // namespace xgboost
