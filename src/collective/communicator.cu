/*!
 * Copyright 2022 XGBoost contributors
 */
#include "communicator.h"
#include "device_communicator.cuh"
#include "device_communicator_adapter.cuh"
#ifdef XGBOOST_USE_NCCL
#include "nccl_device_communicator.cuh"
#endif

namespace xgboost {
namespace collective {

thread_local int Communicator::device_ordinal_{-1};
thread_local std::unique_ptr<DeviceCommunicator> Communicator::device_communicator_{};

void Communicator::Finalize() {
  communicator_.reset();
  device_ordinal_ = -1;
  device_communicator_.reset();
}

DeviceCommunicator* Communicator::GetDevice(int device_ordinal) {
  if (!device_communicator_ || device_ordinal_ != device_ordinal) {
    device_ordinal_ = device_ordinal;
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
