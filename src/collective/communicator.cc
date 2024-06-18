/*!
 * Copyright 2022 XGBoost contributors
 */
#include "communicator.h"

#include "comm.h"
#include "in_memory_communicator.h"
#include "noop_communicator.h"
#include "rabit_communicator.h"

#if defined(XGBOOST_USE_FEDERATED)
#include "../../plugin/federated/federated_communicator.h"
#endif

namespace xgboost::collective {
thread_local std::unique_ptr<Communicator> Communicator::communicator_{new NoOpCommunicator()};
thread_local CommunicatorType Communicator::type_{};
thread_local std::string Communicator::nccl_path_{};

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
}
#endif
}  // namespace xgboost::collective
