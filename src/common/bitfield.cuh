#ifndef XGBOOST_COMMON_BITFIELD_CUH_
#define XGBOOST_COMMON_BITFIELD_CUH_

#include <bitset>
#include <string>
#include <iostream>
#include <sstream>
#include <cinttypes>

#include "span.h"

namespace xgboost {

#if defined(__CUDACC__)
__forceinline__ __device__ unsigned long long AtomicOr(unsigned long long* address,
                                                       unsigned long long val) {
  unsigned long long int old = *address, assumed;  // NOLINT
  do {
    assumed = old;
    old = atomicCAS(address, assumed, val | assumed);
  } while (assumed != old);

  return old;
}

__forceinline__ __device__ unsigned long long AtomicAnd(unsigned long long* address,
                                                        unsigned long long val) {
  unsigned long long int old = *address, assumed;  // NOLINT
  do {
    assumed = old;
    old = atomicCAS(address, assumed, val & assumed);
  } while (assumed != old);

  return old;
}
#endif  // defined(__CUDACC__)

/*!
 * \brief A non-owning type with auxiliary methods defined for manipulating bits.
 */
struct BitField {
  using value_type = uint64_t;
  using pointer = value_type*;

  static value_type constexpr kValueSize = sizeof(value_type) * 8;
  static value_type constexpr kOne = 1UL;  // force uint64_t
  static_assert(kValueSize == 64, "uint64_t should be of 64 bits.");

  struct Pos {
    value_type int_pos {0};
    value_type bit_pos {0};
  };

  common::Span<value_type> bits_;

 public:
  BitField() = default;
  XGBOOST_DEVICE BitField(common::Span<value_type> bits) : bits_{bits} {}
  XGBOOST_DEVICE BitField(BitField const& other) : bits_{other.bits_} {}

  static size_t ComputeStorageSize(size_t size) {
    auto pos = ToBitPos(size);
    if (size < kValueSize) {
      return 1;
    }

    if (pos.bit_pos != 0) {
      return pos.int_pos + 2;
    } else {
      return pos.int_pos + 1;
    }
  }
  XGBOOST_DEVICE static Pos ToBitPos(value_type pos) {
    Pos pos_v;
    if (pos == 0) {
      return pos_v;
    }
    pos_v.int_pos =  pos / kValueSize;
    pos_v.bit_pos =  pos % kValueSize;
    return pos_v;
  }
#if defined(__CUDACC__)
  __device__ BitField& operator|=(BitField const& rhs) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t min_size = min(bits_.size(), rhs.bits_.size());
    if (tid < min_size) {
      bits_[tid] |= rhs.bits_[tid];
    }
    return *this;
  }
  __device__ BitField& operator&=(BitField const& rhs) {
    size_t min_size = min(bits_.size(), rhs.bits_.size());
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < min_size) {
      bits_[tid] &= rhs.bits_[tid];
    }
    return *this;
  }

  __device__ void Set(value_type pos) {
    Pos pos_v = ToBitPos(pos);
    value_type& value = bits_[pos_v.int_pos];
    value_type set_bit = kOne << (kValueSize - pos_v.bit_pos - kOne);
    static_assert(sizeof(unsigned long long int) == sizeof(value_type), "");
    AtomicOr(reinterpret_cast<unsigned long long*>(&value), set_bit);
  }
  __device__ void Clear(value_type pos) {
    Pos pos_v = ToBitPos(pos);
    value_type& value = bits_[pos_v.int_pos];
    value_type clear_bit = ~(kOne << (kValueSize - pos_v.bit_pos - kOne));
    static_assert(sizeof(unsigned long long int) == sizeof(value_type), "");
    AtomicAnd(reinterpret_cast<unsigned long long*>(&value), clear_bit);
  }

  XGBOOST_DEVICE bool Check(Pos pos_v) const {
    value_type value = bits_[pos_v.int_pos];
    value_type const test_bit = kOne << (kValueSize - pos_v.bit_pos - kOne);
    value_type result = test_bit & value;
    return static_cast<bool>(result);
  }
  XGBOOST_DEVICE bool Check(value_type pos) const {
    Pos pos_v = ToBitPos(pos);
    return Check(pos_v);
  }
#endif  // defined(__CUDACC__)

  XGBOOST_DEVICE size_t Size() const { return kValueSize * bits_.size(); }

  XGBOOST_DEVICE pointer Data() const { return bits_.data(); }

  friend std::ostream& operator<<(std::ostream& os, BitField field) {
    os << "Bits " << "storage size: " << field.bits_.size() << "\n";
    for (common::Span<value_type>::index_type i = 0; i < field.bits_.size(); ++i) {
      std::bitset<BitField::kValueSize> set(field.bits_[i]);
      os << set << "\n";
    }
    return os;
  }
};

#if defined(__CUDACC__)

inline void PrintDeviceBits(std::string name, BitField field) {
  std::cout << "Bits: " << name << std::endl;
  std::vector<BitField::value_type> h_field_bits(field.bits_.size());
  thrust::copy(thrust::device_ptr<BitField::value_type>(field.bits_.data()),
               thrust::device_ptr<BitField::value_type>(field.bits_.data() + field.bits_.size()),
               h_field_bits.data());
  BitField h_field;
  h_field.bits_ = {h_field_bits.data(), h_field_bits.data() + h_field_bits.size()};
  std::cout << h_field;
}

inline void PrintDeviceStorage(std::string name, common::Span<int32_t> list) {
  std::cout << name << std::endl;
  std::vector<int32_t> h_list(list.size());
  thrust::copy(thrust::device_ptr<int32_t>(list.data()),
               thrust::device_ptr<int32_t>(list.data() + list.size()),
               h_list.data());
  for (auto v : h_list) {
    std::cout << v << ", ";
  }
  std::cout << std::endl;
}

#endif  // defined(__CUDACC__)
}

#endif  // XGBOOST_COMMON_BITFIELD_CUH_