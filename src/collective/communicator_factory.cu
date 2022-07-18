/*!
 * Copyright 2022 XGBoost contributors
 */
#include <rabit/rabit.h>

#include "communicator_factory.h"
#include "device_communicator_adapter.cuh"
#include "rabit_communicator.h"
#include "nccl_device_communicator.cuh"

namespace xgboost {
namespace collective {

thread_local std::unique_ptr<CommunicatorFactory> CommunicatorFactory::instance_{};

void CommunicatorFactory::Init(int argc, char* argv[]) {
  if (instance_) {
    LOG(FATAL) << "Communicator factory can only be initialized once.";
  }

  auto type = GetTypeFromEnv();
  auto const arg = GetTypeFromArgs(argc, argv);
  if (arg != CommunicatorType::kUnknown) {
    type = arg;
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
      break;
  }
}

void CommunicatorFactory::Finalize() { instance_.reset(); }

CommunicatorFactory::CommunicatorFactory(CommunicatorType type, Communicator* communicator)
    : type_{type}, communicator_{communicator} {}

DeviceCommunicator* CommunicatorFactory::GetDeviceCommunicator(int device_ordinal) {
  if (!device_communicator_) {
#ifdef XGBOOST_USE_NCCL
    if (type_ != CommunicatorType::kFederated) {
      device_communicator_.reset(new NcclDeviceCommunicator(device_ordinal, communicator_.get()));
    } else {
      device_communicator_.reset(
          new DeviceCommunicatorAdapter(device_ordinal, communicator_.get()));
    }
#else
    device_communicator_.reset(new DeviceCommunicatorAdapter(device_ordinal, communicator_.get()));
#endif
  }
  return device_communicator_.get();
}

}  // namespace collective
}  // namespace xgboost
