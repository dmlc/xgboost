/*!
 * Copyright 2022 XGBoost contributors
 */
#include "communicator.h"

#include "in_memory_communicator.h"
#include "noop_communicator.h"
#include "rabit_communicator.h"

#if defined(XGBOOST_USE_FEDERATED)
#include "../../plugin/federated/federated_communicator.h"
#endif

namespace xgboost {
namespace collective {

thread_local std::unique_ptr<Communicator> Communicator::communicator_{new NoOpCommunicator()};
thread_local CommunicatorType Communicator::type_{};

void Communicator::Init(Json const& config) {
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
#else
      LOG(FATAL) << "XGBoost is not compiled with Federated Learning support.";
#endif
      break;
    }
    case CommunicatorType::kInMemory: {
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
}
#endif

}  // namespace collective
}  // namespace xgboost
