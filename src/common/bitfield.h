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
#include "device_helpers.cuh"
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
template <typename VT, typename Direction, bool IsConst = false>
struct BitFieldContainer {
  using value_type = std::conditional_t<IsConst, VT const, VT>;  // NOLINT
  using index_type = size_t;                                     // NOLINT
  using pointer = value_type*;                                   // NOLINT

  static index_type constexpr kValueSize = sizeof(value_type) * 8;
  static index_type constexpr kOne = 1;  // force correct type.

  struct Pos {
    index_type int_pos{0};
    index_type bit_pos{0};
  };

 private:
  common::Span<value_type> bits_;
  static_assert(!std::is_signed<VT>::value, "Must use unsiged type as underlying storage.");

 public:
  XGBOOST_DEVICE static Pos ToBitPos(index_type pos) {
    Pos pos_v;
    if (pos == 0) {
      return pos_v;
    }
    pos_v.int_pos = pos / kValueSize;
    pos_v.bit_pos = pos % kValueSize;
    return pos_v;
  }

 public:
  BitFieldContainer() = default;
  XGBOOST_DEVICE explicit BitFieldContainer(common::Span<value_type> bits) : bits_{bits} {}
  XGBOOST_DEVICE BitFieldContainer(BitFieldContainer const& other) : bits_{other.bits_} {}
  BitFieldContainer &operator=(BitFieldContainer const &that) = default;
  BitFieldContainer &operator=(BitFieldContainer &&that) = default;

  XGBOOST_DEVICE common::Span<value_type>       Bits()       { return bits_; }
  XGBOOST_DEVICE common::Span<value_type const> Bits() const { return bits_; }

  /*\brief Compute the size of needed memory allocation.  The returned value is in terms
   *       of number of elements with `BitFieldContainer::value_type'.
   */
  XGBOOST_DEVICE static size_t ComputeStorageSize(index_type size) {
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
  __device__ auto Set(index_type pos) {
    Pos pos_v = Direction::Shift(ToBitPos(pos));
    value_type& value = bits_[pos_v.int_pos];
    value_type set_bit = kOne << pos_v.bit_pos;
    using Type = typename dh::detail::AtomicDispatcher<sizeof(value_type)>::Type;
    atomicOr(reinterpret_cast<Type *>(&value), set_bit);
  }
  __device__ void Clear(index_type pos) {
    Pos pos_v = Direction::Shift(ToBitPos(pos));
    value_type& value = bits_[pos_v.int_pos];
    value_type clear_bit = ~(kOne << pos_v.bit_pos);
    using Type = typename dh::detail::AtomicDispatcher<sizeof(value_type)>::Type;
    atomicAnd(reinterpret_cast<Type *>(&value), clear_bit);
  }
#else
  void Set(index_type pos) {
    Pos pos_v = Direction::Shift(ToBitPos(pos));
    value_type& value = bits_[pos_v.int_pos];
    value_type set_bit = kOne << pos_v.bit_pos;
    value |= set_bit;
  }
  void Clear(index_type pos) {
    Pos pos_v = Direction::Shift(ToBitPos(pos));
    value_type& value = bits_[pos_v.int_pos];
    value_type clear_bit = ~(kOne << pos_v.bit_pos);
    value &= clear_bit;
  }
#endif  // defined(__CUDA_ARCH__)

  XGBOOST_DEVICE bool Check(Pos pos_v) const {
    pos_v = Direction::Shift(pos_v);
    SPAN_LT(pos_v.int_pos, bits_.size());
    value_type const value = bits_[pos_v.int_pos];
    value_type const test_bit = kOne << pos_v.bit_pos;
    value_type result = test_bit & value;
    return static_cast<bool>(result);
  }
  XGBOOST_DEVICE bool Check(index_type pos) const {
    Pos pos_v = ToBitPos(pos);
    return Check(pos_v);
  }

  XGBOOST_DEVICE size_t Size() const { return kValueSize * bits_.size(); }

  XGBOOST_DEVICE pointer Data() const { return bits_.data(); }

  inline friend std::ostream &
  operator<<(std::ostream &os, BitFieldContainer<VT, Direction, IsConst> field) {
    os << "Bits " << "storage size: " << field.bits_.size() << "\n";
    for (typename common::Span<value_type>::index_type i = 0; i < field.bits_.size(); ++i) {
      std::bitset<BitFieldContainer<VT, Direction, IsConst>::kValueSize> bset(field.bits_[i]);
      os << bset << "\n";
    }
    return os;
  }
};

// Bits start from left most bits (most significant bit).
template <typename VT, bool IsConst = false>
struct LBitsPolicy : public BitFieldContainer<VT, LBitsPolicy<VT, IsConst>, IsConst> {
  using Container = BitFieldContainer<VT, LBitsPolicy<VT, IsConst>, IsConst>;
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

// Format: <Const><Direction>BitField<size of underlying type in bits>, underlying type
// must be unsigned.
using LBitField64 = BitFieldContainer<uint64_t, LBitsPolicy<uint64_t>>;
using RBitField8 = BitFieldContainer<uint8_t, RBitsPolicy<unsigned char>>;

using LBitField32 = BitFieldContainer<uint32_t, LBitsPolicy<uint32_t>>;
using CLBitField32 = BitFieldContainer<uint32_t, LBitsPolicy<uint32_t, true>, true>;
}       // namespace xgboost

#endif  // XGBOOST_COMMON_BITFIELD_H_
