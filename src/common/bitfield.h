/**
 * Copyright 2019-2023, XGBoost Contributors
 * \file bitfield.h
 */
#ifndef XGBOOST_COMMON_BITFIELD_H_
#define XGBOOST_COMMON_BITFIELD_H_

#include <algorithm>    // for min
#include <bitset>       // for bitset
#include <cstdint>      // for uint32_t, uint64_t, uint8_t
#include <ostream>      // for ostream
#include <type_traits>  // for conditional_t, is_signed_v, add_const_t

#if defined(__CUDACC__)
#include <thrust/copy.h>
#include <thrust/device_ptr.h>

#include "device_helpers.cuh"
#endif  // defined(__CUDACC__)

#include "common.h"
#include "xgboost/span.h"  // for Span

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

/**
 * @brief A non-owning type with auxiliary methods defined for manipulating bits.
 *
 * @tparam VT        Underlying value type, must be an unsigned integer.
 * @tparam Direction Whether the bits start from left or from right.
 * @tparam IsConst   Whether the view is const.
 */
template <typename VT, typename Direction, bool IsConst = false>
struct BitFieldContainer {
  using value_type = std::conditional_t<IsConst, VT const, VT>;  // NOLINT
  using size_type = size_t;                                      // NOLINT
  using index_type = size_t;                                     // NOLINT
  using pointer = value_type*;                                   // NOLINT

  static index_type constexpr kValueSize = sizeof(value_type) * 8;
  static index_type constexpr kOne = 1;  // force correct type.

  struct Pos {
    index_type int_pos{0};
    index_type bit_pos{0};
  };

 private:
  value_type* bits_{nullptr};
  size_type n_values_{0};
  static_assert(!std::is_signed_v<VT>, "Must use an unsiged type as the underlying storage.");

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
  XGBOOST_DEVICE explicit BitFieldContainer(common::Span<value_type> bits)
      : bits_{bits.data()}, n_values_{bits.size()} {}
  BitFieldContainer(BitFieldContainer const& other) = default;
  BitFieldContainer(BitFieldContainer&& other) = default;
  BitFieldContainer &operator=(BitFieldContainer const &that) = default;
  BitFieldContainer &operator=(BitFieldContainer &&that) = default;

  XGBOOST_DEVICE auto Bits() { return common::Span<value_type>{bits_, NumValues()}; }
  XGBOOST_DEVICE auto Bits() const { return common::Span<value_type const>{bits_, NumValues()}; }

  /*\brief Compute the size of needed memory allocation.  The returned value is in terms
   *       of number of elements with `BitFieldContainer::value_type'.
   */
  XGBOOST_DEVICE static size_t ComputeStorageSize(index_type size) {
    return common::DivRoundUp(size, kValueSize);
  }
#if defined(__CUDA_ARCH__)
  __device__ BitFieldContainer& operator|=(BitFieldContainer const& rhs) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t min_size = std::min(this->Capacity(), rhs.Capacity());
    if (tid < min_size) {
      if (this->Check(tid) || rhs.Check(tid)) {
        this->Set(tid);
      }
    }
    return *this;
  }
#else
  BitFieldContainer& operator|=(BitFieldContainer const& rhs) {
    size_t min_size = std::min(NumValues(), rhs.NumValues());
    for (size_t i = 0; i < min_size; ++i) {
      Data()[i] |= rhs.Data()[i];
    }
    return *this;
  }
#endif  // #if defined(__CUDA_ARCH__)

#if defined(__CUDA_ARCH__)
  __device__ BitFieldContainer& operator&=(BitFieldContainer const& rhs) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t min_size = std::min(this->Capacity(), rhs.Capacity());
    if (tid < min_size) {
      if (this->Check(tid) && rhs.Check(tid)) {
        this->Set(tid);
      } else {
        this->Clear(tid);
      }
    }
    return *this;
  }
#else
  BitFieldContainer& operator&=(BitFieldContainer const& rhs) {
    std::size_t min_size = std::min(NumValues(), rhs.NumValues());
    for (size_t i = 0; i < min_size; ++i) {
      Data()[i] &= rhs.Data()[i];
    }
    return *this;
  }
#endif  // defined(__CUDA_ARCH__)

#if defined(__CUDA_ARCH__)
  __device__ auto Set(index_type pos) noexcept(true) {
    Pos pos_v = Direction::Shift(ToBitPos(pos));
    value_type& value = Data()[pos_v.int_pos];
    value_type set_bit = kOne << pos_v.bit_pos;
    using Type = typename dh::detail::AtomicDispatcher<sizeof(value_type)>::Type;
    atomicOr(reinterpret_cast<Type *>(&value), set_bit);
  }
  __device__ void Clear(index_type pos) noexcept(true) {
    Pos pos_v = Direction::Shift(ToBitPos(pos));
    value_type& value = Data()[pos_v.int_pos];
    value_type clear_bit = ~(kOne << pos_v.bit_pos);
    using Type = typename dh::detail::AtomicDispatcher<sizeof(value_type)>::Type;
    atomicAnd(reinterpret_cast<Type *>(&value), clear_bit);
  }
#else
  void Set(index_type pos) noexcept(true) {
    Pos pos_v = Direction::Shift(ToBitPos(pos));
    value_type& value = Data()[pos_v.int_pos];
    value_type set_bit = kOne << pos_v.bit_pos;
    value |= set_bit;
  }
  void Clear(index_type pos) noexcept(true) {
    Pos pos_v = Direction::Shift(ToBitPos(pos));
    value_type& value = Data()[pos_v.int_pos];
    value_type clear_bit = ~(kOne << pos_v.bit_pos);
    value &= clear_bit;
  }
#endif  // defined(__CUDA_ARCH__)

  XGBOOST_DEVICE bool Check(Pos pos_v) const noexcept(true) {
    pos_v = Direction::Shift(pos_v);
    assert(pos_v.int_pos < NumValues());
    value_type const value = Data()[pos_v.int_pos];
    value_type const test_bit = kOne << pos_v.bit_pos;
    value_type result = test_bit & value;
    return static_cast<bool>(result);
  }
  [[nodiscard]] XGBOOST_DEVICE bool Check(index_type pos) const noexcept(true) {
    Pos pos_v = ToBitPos(pos);
    return Check(pos_v);
  }
  /**
   * @brief Returns the total number of bits that can be viewed. This is equal to or
   *        larger than the acutal number of valid bits.
   */
  [[nodiscard]] XGBOOST_DEVICE size_type Capacity() const noexcept(true) {
    return kValueSize * NumValues();
  }
  /**
   * @brief Number of storage unit used in this bit field.
   */
  [[nodiscard]] XGBOOST_DEVICE size_type NumValues() const noexcept(true) { return n_values_; }

  XGBOOST_DEVICE pointer Data() const noexcept(true) { return bits_; }

  inline friend std::ostream& operator<<(std::ostream& os,
                                         BitFieldContainer<VT, Direction, IsConst> field) {
    os << "Bits "
       << "storage size: " << field.NumValues() << "\n";
    for (typename common::Span<value_type>::index_type i = 0; i < field.NumValues(); ++i) {
      std::bitset<BitFieldContainer<VT, Direction, IsConst>::kValueSize> bset(field.Data()[i]);
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
using LBitField64 = BitFieldContainer<std::uint64_t, LBitsPolicy<std::uint64_t>>;
using RBitField8 = BitFieldContainer<std::uint8_t, RBitsPolicy<unsigned char>>;

using LBitField32 = BitFieldContainer<std::uint32_t, LBitsPolicy<std::uint32_t>>;
using CLBitField32 = BitFieldContainer<std::uint32_t, LBitsPolicy<std::uint32_t, true>, true>;
using RBitField32 = BitFieldContainer<std::uint32_t, RBitsPolicy<std::uint32_t>>;

namespace detail {
inline std::uint32_t TrailingZeroBitsImpl(std::uint32_t value) {
  auto n = sizeof(value) * 8;
  std::uint32_t cnt{0};
  for (decltype(n) i = 0; i < n; i++) {
    if ((value >> i) & 1) {
      break;
    }
    cnt++;
  }
  return cnt;
}
}  // namespace detail

inline std::uint32_t TrailingZeroBits(std::uint32_t value) {
  if (value == 0) {
    return sizeof(value) * 8;
  }
#if defined(__GNUC__)
  return __builtin_ctz(value);
#elif defined(_MSC_VER)
  return _tzcnt_u32(value);
#else
  return detail::TrailingZeroBitsImpl(value);
#endif  //  __GNUC__
}
}       // namespace xgboost

#endif  // XGBOOST_COMMON_BITFIELD_H_
