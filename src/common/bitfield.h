/*!
 * Copyright 2019 by Contributors
 * \file bitfield.h
 */
#ifndef XGBOOST_COMMON_BITFIELD_H_
#define XGBOOST_COMMON_BITFIELD_H_

#include <algorithm>
#include <bitset>
#include <cinttypes>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined(__CUDACC__)
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#endif  // defined(__CUDACC__)

#include "xgboost/span.h"
#include "common.h"

namespace xgboost {

#if defined(__CUDACC__)
using BitFieldAtomicType = unsigned long long;  // NOLINT

__forceinline__ __device__ BitFieldAtomicType AtomicOr(BitFieldAtomicType* address,
                                                       BitFieldAtomicType val) {
  BitFieldAtomicType old = *address, assumed;  // NOLINT
  do {
    assumed = old;
    old = atomicCAS(address, assumed, val | assumed);
  } while (assumed != old);

  return old;
}

__forceinline__ __device__ BitFieldAtomicType AtomicAnd(BitFieldAtomicType* address,
                                                        BitFieldAtomicType val) {
  BitFieldAtomicType old = *address, assumed;  // NOLINT
  do {
    assumed = old;
    old = atomicCAS(address, assumed, val & assumed);
  } while (assumed != old);

  return old;
}
#endif  // defined(__CUDACC__)

/*!
 * \brief A non-owning type with auxiliary methods defined for manipulating bits.
 *
 * \tparam Direction Whether the bits start from left or from right.
 */
template <typename VT, typename Direction>
struct BitFieldContainer {
  using value_type = VT;  // NOLINT
  using pointer = value_type*;  // NOLINT

  static value_type constexpr kValueSize = sizeof(value_type) * 8;
  static value_type constexpr kOne = 1;  // force correct type.

  struct Pos {
    value_type int_pos {0};
    value_type bit_pos {0};
  };

 private:
  common::Span<value_type> bits_;
  static_assert(!std::is_signed<VT>::value, "Must use unsiged type as underlying storage.");

  XGBOOST_DEVICE static Pos ToBitPos(value_type pos) {
    Pos pos_v;
    if (pos == 0) {
      return pos_v;
    }
    pos_v.int_pos =  pos / kValueSize;
    pos_v.bit_pos =  pos % kValueSize;
    return pos_v;
  }

 public:
  BitFieldContainer() = default;
  XGBOOST_DEVICE explicit BitFieldContainer(common::Span<value_type> bits) : bits_{bits} {}
  XGBOOST_DEVICE BitFieldContainer(BitFieldContainer const& other) : bits_{other.bits_} {}

  common::Span<value_type>       Bits()       { return bits_; }
  common::Span<value_type const> Bits() const { return bits_; }

  /*\brief Compute the size of needed memory allocation.  The returned value is in terms
   *       of number of elements with `BitFieldContainer::value_type'.
   */
  static size_t ComputeStorageSize(size_t size) {
    return common::DivRoundUp(size, kValueSize);
  }
#if defined(__CUDA_ARCH__)
  __device__ BitFieldContainer& operator|=(BitFieldContainer const& rhs) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t min_size = min(bits_.size(), rhs.bits_.size());
    if (tid < min_size) {
      bits_[tid] |= rhs.bits_[tid];
    }
    return *this;
  }
#else
  BitFieldContainer& operator|=(BitFieldContainer const& rhs) {
    size_t min_size = std::min(bits_.size(), rhs.bits_.size());
    for (size_t i = 0; i < min_size; ++i) {
      bits_[i] |= rhs.bits_[i];
    }
    return *this;
  }
#endif  // #if defined(__CUDA_ARCH__)

#if defined(__CUDA_ARCH__)
  __device__ BitFieldContainer& operator&=(BitFieldContainer const& rhs) {
    size_t min_size = min(bits_.size(), rhs.bits_.size());
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < min_size) {
      bits_[tid] &= rhs.bits_[tid];
    }
    return *this;
  }
#else
  BitFieldContainer& operator&=(BitFieldContainer const& rhs) {
    size_t min_size = std::min(bits_.size(), rhs.bits_.size());
    for (size_t i = 0; i < min_size; ++i) {
      bits_[i] &= rhs.bits_[i];
    }
    return *this;
  }
#endif  // defined(__CUDA_ARCH__)

#if defined(__CUDA_ARCH__)
  __device__ void Set(value_type pos) {
    Pos pos_v = Direction::Shift(ToBitPos(pos));
    value_type& value = bits_[pos_v.int_pos];
    value_type set_bit = kOne << pos_v.bit_pos;
    static_assert(sizeof(BitFieldAtomicType) == sizeof(value_type), "");
    AtomicOr(reinterpret_cast<BitFieldAtomicType*>(&value), set_bit);
  }
  __device__ void Clear(value_type pos) {
    Pos pos_v = Direction::Shift(ToBitPos(pos));
    value_type& value = bits_[pos_v.int_pos];
    value_type clear_bit = ~(kOne << pos_v.bit_pos);
    static_assert(sizeof(BitFieldAtomicType) == sizeof(value_type), "");
    AtomicAnd(reinterpret_cast<BitFieldAtomicType*>(&value), clear_bit);
  }
#else
  void Set(value_type pos) {
    Pos pos_v = Direction::Shift(ToBitPos(pos));
    value_type& value = bits_[pos_v.int_pos];
    value_type set_bit = kOne << pos_v.bit_pos;
    value |= set_bit;
  }
  void Clear(value_type pos) {
    Pos pos_v = Direction::Shift(ToBitPos(pos));
    value_type& value = bits_[pos_v.int_pos];
    value_type clear_bit = ~(kOne << pos_v.bit_pos);
    value &= clear_bit;
  }
#endif  // defined(__CUDA_ARCH__)

  XGBOOST_DEVICE bool Check(Pos pos_v) const {
    pos_v = Direction::Shift(pos_v);
    value_type const value = bits_[pos_v.int_pos];
    value_type const test_bit = kOne << pos_v.bit_pos;
    value_type result = test_bit & value;
    return static_cast<bool>(result);
  }
  XGBOOST_DEVICE bool Check(value_type pos) const {
    Pos pos_v = ToBitPos(pos);
    return Check(pos_v);
  }

  XGBOOST_DEVICE size_t Size() const { return kValueSize * bits_.size(); }

  XGBOOST_DEVICE pointer Data() const { return bits_.data(); }

  friend std::ostream& operator<<(std::ostream& os, BitFieldContainer<VT, Direction> field) {
    os << "Bits " << "storage size: " << field.bits_.size() << "\n";
    for (typename common::Span<value_type>::index_type i = 0; i < field.bits_.size(); ++i) {
      std::bitset<BitFieldContainer<VT, Direction>::kValueSize> bset(field.bits_[i]);
      os << bset << "\n";
    }
    return os;
  }
};

// Bits start from left most bits (most significant bit).
template <typename VT>
struct LBitsPolicy : public BitFieldContainer<VT, LBitsPolicy<VT>> {
  using Container = BitFieldContainer<VT, LBitsPolicy<VT>>;
  using Pos = typename Container::Pos;
  using value_type = typename Container::value_type;  // NOLINT

  XGBOOST_DEVICE static Pos Shift(Pos pos) {
    pos.bit_pos = Container::kValueSize - pos.bit_pos - Container::kOne;
    return pos;
  }
};

// Bits start from right most bit (least significant bit) of each entry, but integer index
// is from left to right.
template <typename VT>
struct RBitsPolicy : public BitFieldContainer<VT, RBitsPolicy<VT>> {
  using Container = BitFieldContainer<VT, RBitsPolicy<VT>>;
  using Pos = typename Container::Pos;
  using value_type = typename Container::value_type;  // NOLINT

  XGBOOST_DEVICE static Pos Shift(Pos pos) {
    return pos;
  }
};

// Format: <Direction>BitField<size of underlying type in bits>, underlying type must be unsigned.
using LBitField64 = BitFieldContainer<uint64_t, LBitsPolicy<uint64_t>>;
using RBitField8 = BitFieldContainer<uint8_t, RBitsPolicy<unsigned char>>;

#if defined(__CUDACC__)

template <typename V, typename D>
inline void PrintDeviceBits(std::string name, BitFieldContainer<V, D> field) {
  std::cout << "Bits: " << name << std::endl;
  std::vector<typename BitFieldContainer<V, D>::value_type> h_field_bits(field.bits_.size());
  thrust::copy(thrust::device_ptr<typename BitFieldContainer<V, D>::value_type>(field.bits_.data()),
               thrust::device_ptr<typename BitFieldContainer<V, D>::value_type>(
                   field.bits_.data() + field.bits_.size()),
               h_field_bits.data());
  BitFieldContainer<V, D> h_field;
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
}       // namespace xgboost

#endif  // XGBOOST_COMMON_BITFIELD_H_
