/*!
 * Copyright 2022-2024 XGBoost contributors
 */
#pragma once

#include "../helpers.h"

namespace xgboost::sycl {
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
