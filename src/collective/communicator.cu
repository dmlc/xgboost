/*!
 * Copyright 2022 XGBoost contributors
 */
#include "communicator.h"
#include "device_communicator.cuh"
#include "device_communicator_adapter.cuh"
#include "noop_communicator.h"
#ifdef XGBOOST_USE_NCCL
#include "nccl_device_communicator.cuh"
#endif

namespace xgboost {
namespace collective {

thread_local std::unique_ptr<DeviceCommunicator> Communicator::device_communicator_{};

void Communicator::Finalize() {
  communicator_->Shutdown();
  communicator_.reset(new NoOpCommunicator());
  device_communicator_.reset(nullptr);
}

DeviceCommunicator* Communicator::GetDevice(int device_ordinal) {
  thread_local auto old_device_ordinal = -1;
  // If the number of GPUs changes, we need to re-initialize NCCL.
  thread_local auto old_world_size = -1;
  if (!device_communicator_ || device_ordinal != old_device_ordinal ||
      communicator_->GetWorldSize() != old_world_size) {
    old_device_ordinal = device_ordinal;
    old_world_size = communicator_->GetWorldSize();
#ifdef XGBOOST_USE_NCCL
    if (type_ != CommunicatorType::kFederated) {
      device_communicator_.reset(new NcclDeviceCommunicator(device_ordinal, Get()));
    } else {
      device_communicator_.reset(new DeviceCommunicatorAdapter(device_ordinal, Get()));
    }
#else
    device_communicator_.reset(new DeviceCommunicatorAdapter(device_ordinal, Get()));
#endif
  }
  return device_communicator_.get();
}

}  // namespace collective
}  // namespace xgboost
