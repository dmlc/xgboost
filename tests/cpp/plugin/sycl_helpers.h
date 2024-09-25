/*!
 * Copyright 2022-2024 XGBoost contributors
 */
#pragma once

#include "../helpers.h"
#include "../../plugin/sycl/device_manager.h"
#include "../../plugin/sycl/data.h"

namespace xgboost::sycl {

template<typename T, typename Fn>
void TransformOnDeviceData(DeviceOrd device, T* device_data, size_t n_data, Fn&& fn) {
  sycl::DeviceManager device_manager;
  ::sycl::queue* qu = device_manager.GetQueue(device);

  qu->submit([&](::sycl::handler& cgh) {
    cgh.parallel_for<>(::sycl::range<1>(n_data), [=](::sycl::item<1> nid) {
      const size_t i = nid.get_id(0);
      device_data[i] = fn(device_data[i]);
    });
  }).wait();
}

template<typename T>
void VerifyOnDeviceData(DeviceOrd device, const T* device_data, const T* host_data, size_t n_data, T eps = T()) {
  sycl::DeviceManager device_manager;
  ::sycl::queue* qu = device_manager.GetQueue(device);

  std::vector<T> copy_device_data(n_data);
  qu->memcpy(copy_device_data.data(), device_data, n_data * sizeof(T)).wait();
  for (size_t i = 0; i < n_data; ++i) {
    EXPECT_NEAR(copy_device_data[i], host_data[i], eps);
  }
}

template<typename T, typename Container>
void VerifySyclVector(const USMVector<T, MemoryType::shared>& sycl_vector,
                      const Container& host_vector, T eps = T()) {
  ASSERT_EQ(sycl_vector.Size(), host_vector.size());

  size_t size = sycl_vector.Size();
  for (size_t i = 0; i < size; ++i) {
    EXPECT_NEAR(sycl_vector[i], host_vector[i], eps);
  }
}

template<typename T, typename Container>
void VerifySyclVector(const std::vector<T>& sycl_vector,
                      const Container& host_vector, T eps = T()) {
  ASSERT_EQ(sycl_vector.size(), host_vector.size());

  size_t size = sycl_vector.size();
  for (size_t i = 0; i < size; ++i) {
    EXPECT_NEAR(sycl_vector[i], host_vector[i], eps);
  }
}

}  // namespace xgboost::sycl
