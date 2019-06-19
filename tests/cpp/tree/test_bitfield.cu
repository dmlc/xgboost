/*!
 * Copyright 2019 XGBoost contributors
 */
#include <gtest/gtest.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <vector>
#include "../../../src/tree/constraints.cuh"
#include "../../../src/common/device_helpers.cuh"

namespace xgboost {

__global__ void TestSetKernel(BitField bits) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < bits.Size()) {
    bits.Set(tid);
  }
}

TEST(BitField, Set) {
  dh::device_vector<BitField::value_type> storage;
  uint32_t constexpr kBits = 128;
  storage.resize(128);
  auto bits = BitField(dh::ToSpan(storage));
  TestSetKernel<<<1, kBits>>>(bits);

  std::vector<BitField::value_type> h_storage(storage.size());
  thrust::copy(storage.begin(), storage.end(), h_storage.begin());

  BitField outputs {
    common::Span<BitField::value_type>{h_storage.data(),
                                       h_storage.data() + h_storage.size()}};
  for (size_t i = 0; i < kBits; ++i) {
    ASSERT_TRUE(outputs.Check(i));
  }
}

__global__ void TestOrKernel(BitField lhs, BitField rhs) {
  lhs |= rhs;
}

TEST(BitField, And) {
  uint32_t constexpr kBits = 128;
  dh::device_vector<BitField::value_type> lhs_storage(kBits);
  dh::device_vector<BitField::value_type> rhs_storage(kBits);
  auto lhs = BitField(dh::ToSpan(lhs_storage));
  auto rhs = BitField(dh::ToSpan(rhs_storage));
  thrust::fill(lhs_storage.begin(), lhs_storage.end(), 0UL);
  thrust::fill(rhs_storage.begin(), rhs_storage.end(), ~static_cast<BitField::value_type>(0UL));
  TestOrKernel<<<1, kBits>>>(lhs, rhs);

  std::vector<BitField::value_type> h_storage(lhs_storage.size());
  thrust::copy(lhs_storage.begin(), lhs_storage.end(), h_storage.begin());
  BitField outputs {{h_storage.data(), h_storage.data() + h_storage.size()}};
  for (size_t i = 0; i < kBits; ++i) {
    ASSERT_TRUE(outputs.Check(i));
  }
}

}  // namespace xgboost