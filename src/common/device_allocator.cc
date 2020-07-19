/*!
 * Copyright 2020 by XGBoost Contributors
 * \file device_allocator.cc
 * \brief Store callback functions for allocating and de-allocating memory on GPU devices.
 */
#include <mutex>
#include "device_allocator.h"

namespace dh {
namespace detail {

DeviceMemoryResource DeviceMemoryResourceSingleton{nullptr, nullptr};
std::mutex DeviceMemoryResourceSingletonMutex;

}  // namespace detail
}  // namespace dh
