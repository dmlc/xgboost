/**
 * Copyright 2017-2025, XGBoost Contributors
 * \file compressed_iterator.h
 */
#pragma once
#include <algorithm>  // for max
#include <cmath>      // for ceil, log2
#include <cstddef>    // for size_t
#include <cstdint>    // for uint32_t

#include "common.h"
#include "xgboost/base.h"      // for XGBOOST_RESTRICT
#include "xgboost/byteswap.h"  // for ByteSwap

#ifdef __CUDACC__
#include "device_helpers.cuh"
#endif  // __CUDACC__

namespace xgboost::common {
using CompressedByteT = unsigned char;

namespace detail {
inline void SetBit(CompressedByteT *byte, int bit_idx) { *byte |= 1 << bit_idx; }
template <typename T>
inline T CheckBit(const T &byte, int bit_idx) {
  return byte & (1 << bit_idx);
}
inline void ClearBit(CompressedByteT *byte, int bit_idx) { *byte &= ~(1 << bit_idx); }
inline constexpr int kPadding = 8;  // Assign padding so we can read slightly off
                                    // the beginning of the array

// The number of bits required to represent a given unsigned range
inline XGBOOST_DEVICE std::uint32_t SymbolBits(std::size_t n_symbols) {
  std::uint32_t bits = std::ceil(log2(static_cast<double>(n_symbols)));
  return common::Max(bits, std::uint32_t{1});
}

// The alignment is assumed to be power of 2.
template <typename T>
XGBOOST_HOST_DEV_INLINE CompressedByteT const *AlignDown(T const *ptr, std::uint32_t alignment) {
  return reinterpret_cast<CompressedByteT const *>(reinterpret_cast<std::uintptr_t>(ptr) &
                                                   ~std::uintptr_t{alignment - 1});
}

struct PaddedPtr {
  CompressedByteT const *XGBOOST_RESTRICT ptr;
  std::int32_t head_padding;
};

// Create an aligned pointer with head padding.
template <typename T>
XGBOOST_DEVICE auto MakePaddedPtr(T const *XGBOOST_RESTRICT ptr, std::uint32_t alignment) {
  auto base = AlignDown(ptr, alignment);
  return PaddedPtr{
      base, static_cast<std::int32_t>(reinterpret_cast<CompressedByteT const *>(ptr) - base)};
}

// Vector load, load a single 64-bit unsigned integer with 2 32-bit loads. Input ptr must
// be correctly aligned first.
template <typename T>
XGBOOST_DEVICE [[nodiscard]] std::uint64_t Load64u(T const *XGBOOST_RESTRICT ptr) {
  std::uint64_t u64 = 0;
  auto out_ptr = reinterpret_cast<std::uint32_t *>(&u64);
  // base ptr in uint32
  auto in_ptr = reinterpret_cast<std::uint32_t const *>(ptr);
  // 2 vector loads for 8 bytes.
  out_ptr[0] = in_ptr[0];
  out_ptr[1] = in_ptr[1];
  return u64;
}
}  // namespace detail

/**
 * \class CompressedBufferWriter
 *
 * \brief Writes bit compressed symbols to a memory buffer. Use
 * CompressedIterator to read symbols back from buffer. Currently limited to a
 * maximum symbol size of 28 bits.
 *
 * \author  Rory
 * \date  7/9/2017
 */

class CompressedBufferWriter {
  std::size_t symbol_bits_;

 public:
  XGBOOST_DEVICE explicit CompressedBufferWriter(std::size_t num_symbols)
      : symbol_bits_{detail::SymbolBits(num_symbols)} {}

  /**
   * \fn  static size_t CompressedBufferWriter::CalculateBufferSize(int
   * num_elements, int num_symbols)
   *
   * \brief Calculates number of bytes required for a given number of elements
   * and a symbol range.
   *
   * \author  Rory
   * \date  7/9/2017
   *
   * \param num_elements  Number of elements.
   * \param num_symbols   Max number of symbols (alphabet size)
   *
   * \return  The calculated buffer size.
   */
  static size_t CalculateBufferSize(size_t num_elements, size_t num_symbols) {
    constexpr int kBitsPerByte = 8;
    size_t compressed_size = static_cast<size_t>(std::ceil(
        static_cast<double>(detail::SymbolBits(num_symbols) * num_elements) / kBitsPerByte));
    // Handle atomicOr where input must be unsigned int, hence 4 bytes aligned.
    size_t ret = std::ceil(static_cast<double>(compressed_size + detail::kPadding) /
                           static_cast<double>(sizeof(std::uint32_t))) *
                 sizeof(std::uint32_t);
    // Need at least 5 bytes for the reader
    return std::max(ret, static_cast<std::size_t>(detail::kPadding + 1));
  }

  template <typename T>
  void WriteSymbol(CompressedByteT *buffer, T symbol, size_t offset) {
    constexpr std::int32_t kBitsPerByte = 8;

    for (size_t i = 0; i < symbol_bits_; i++) {
      size_t byte_idx = ((offset + 1) * symbol_bits_ - (i + 1)) / kBitsPerByte;
      byte_idx += detail::kPadding;
      size_t bit_idx = ((kBitsPerByte + i) - ((offset + 1) * symbol_bits_)) % kBitsPerByte;

      if (detail::CheckBit(symbol, i)) {
        detail::SetBit(&buffer[byte_idx], bit_idx);
      } else {
        detail::ClearBit(&buffer[byte_idx], bit_idx);
      }
    }
  }

#ifdef __CUDACC__
  __device__ void AtomicWriteSymbol(CompressedByteT *buffer, uint64_t symbol, size_t offset) {
    size_t ibit_start = offset * symbol_bits_;
    size_t ibit_end = (offset + 1) * symbol_bits_ - 1;
    size_t ibyte_start = ibit_start / 8, ibyte_end = ibit_end / 8;

    symbol <<= 7 - ibit_end % 8;
    for (ptrdiff_t ibyte = ibyte_end; ibyte >= static_cast<ptrdiff_t>(ibyte_start); --ibyte) {
      dh::AtomicOrByte(reinterpret_cast<unsigned int *>(buffer + detail::kPadding), ibyte,
                       symbol & 0xff);
      symbol >>= 8;
    }
  }
#endif  // __CUDACC__

  template <typename IterT>
  void Write(CompressedByteT *buffer, IterT input_begin, IterT input_end) {
    uint64_t tmp = 0;
    size_t stored_bits = 0;
    const size_t max_stored_bits = 64 - symbol_bits_;
    size_t buffer_position = detail::kPadding;
    const size_t num_symbols = input_end - input_begin;
    for (size_t i = 0; i < num_symbols; i++) {
      typename std::iterator_traits<IterT>::value_type symbol = input_begin[i];
      if (stored_bits > max_stored_bits) {
        // Eject only full bytes
        size_t tmp_bytes = stored_bits / 8;
        for (size_t j = 0; j < tmp_bytes; j++) {
          buffer[buffer_position] =
              static_cast<CompressedByteT>(tmp >> (stored_bits - (j + 1) * 8));
          buffer_position++;
        }
        stored_bits -= tmp_bytes * 8;
        tmp &= (1 << stored_bits) - 1;
      }
      // Store symbol
      tmp <<= symbol_bits_;
      tmp |= symbol;
      stored_bits += symbol_bits_;
    }

    // Eject all bytes
    int tmp_bytes = static_cast<int>(std::ceil(static_cast<float>(stored_bits) / 8));
    for (int j = 0; j < tmp_bytes; j++) {
      int shift_bits = static_cast<int>(stored_bits) - (j + 1) * 8;
      if (shift_bits >= 0) {
        buffer[buffer_position] = static_cast<CompressedByteT>(tmp >> shift_bits);
      } else {
        buffer[buffer_position] = static_cast<CompressedByteT>(tmp << std::abs(shift_bits));
      }
      buffer_position++;
    }
  }
};

/**
 * @brief Read symbols from a bit compressed memory buffer. Usable on device and host.
 *
 * @author  Rory
 *
 * @tparam  T          Type of the symbols.
 */
template <typename T>
class CompressedIterator {
 public:
  typedef T value_type;          // NOLINT
  typedef value_type *pointer;   // NOLINT
  typedef value_type reference;  // NOLINT

 private:
  CompressedByteT const *XGBOOST_RESTRICT buffer_{nullptr};
  std::uint32_t const symbol_bits_{0};

  static_assert(sizeof(T) <= sizeof(std::uint32_t));
  static_assert(detail::kPadding >= std::alignment_of_v<std::uint32_t>);

 public:
  CompressedIterator() = default;
  CompressedIterator(CompressedByteT const *XGBOOST_RESTRICT buffer, bst_idx_t n_symbols)
      : buffer_{buffer}, symbol_bits_{detail::SymbolBits(n_symbols)} {
#if !defined(DMLC_LITTLE_ENDIAN) || DMLC_LITTLE_ENDIAN != 1
    LOG(FATAL) << "Not implemented for big endian";
#endif
  }

  XGBOOST_DEVICE reference operator[](std::size_t idx) const {
    constexpr std::int32_t kBitsPerByte = 8;
    // Read 5 bytes - the maximum we will need assuming symbols fit in a 32bit int.
    constexpr std::int32_t kBytes = 5;

    std::size_t start_bit_idx = ((idx + 1) * symbol_bits_ - 1);
    std::size_t start_byte_idx = start_bit_idx / kBitsPerByte;
    start_byte_idx += detail::kPadding;

    /**
     * The following load is equivalent to:
     *
     * std::uint64_t tmp = static_cast<std::uint64_t>(buffer_[start_byte_idx - 4]) << 32 |
     *                     static_cast<std::uint64_t>(buffer_[start_byte_idx - 3]) << 24 |
     *                     static_cast<std::uint64_t>(buffer_[start_byte_idx - 2]) << 16 |
     *                     static_cast<std::uint64_t>(buffer_[start_byte_idx - 1]) << 8 |
     *                     buffer_[start_byte_idx];
     *
     * The above snippet loads 5 bytes from the buffer, and performs a byte swap within
     * the loaded 5 bytes. We use a vector load to reduce the pressure on the LSU.
     */

    // Pointer to the first byte.
    auto beg_ptr = buffer_ + start_byte_idx - (kBytes - 1);
    // Align the pointer for vector load.
    auto [ptr, head_padding] = detail::MakePaddedPtr(beg_ptr, std::alignment_of_v<std::uint32_t>);
    // Load 8 bytes, we will use 5 of them.
    std::uint64_t tmp = detail::Load64u(ptr);
    // tail_padding = 8 - 5 - head_padding
    std::int32_t tail_padding_bits = (sizeof(tmp) - kBytes - head_padding) * kBitsPerByte;
    // Unsigned logical shift. Knock out the unneeded bits loaded by the vector load. We
    // assume little endian here.
    tmp = ByteSwap(tmp << tail_padding_bits);

    // Knock out the unneeded bits from the right
    std::int32_t bit_shift = (kBitsPerByte - ((idx + 1) * symbol_bits_)) % kBitsPerByte;
    tmp >>= bit_shift;
    // Take exactly symbol_bits_ number of bits by masking off unneeded bits.
    std::uint64_t mask = (static_cast<std::uint64_t>(1) << symbol_bits_) - 1;
    return static_cast<T>(tmp & mask);
  }
};

/**
 * @brief A compressed iterator with two buffers for the underlying storage.
 *
 * This accessor is significantly slower than the single buffer one due to pipeline
 * stalling and should not be used as default. Pre-calculating the buffer selection
 * indicator can help mitigate it. But we only use this iterator for external memory with
 * direct memory access, which is slow anyway.
 *
 * Use the single buffer one as a reference for how it works.
 */
template <typename OutT>
class DoubleCompressedIter {
 public:
  using value_type = OutT;       // NOLINT
  using pointer = value_type *;  // NOLINT
  using reference = value_type;  // NOLINT

 private:
  using BufT = CompressedByteT const *;
  BufT XGBOOST_RESTRICT buf0_{nullptr};
  BufT XGBOOST_RESTRICT buf1_{nullptr};
  bst_idx_t const n0_{0};  // Size of the first buffer in bytes.
  std::uint32_t const symbol_bits_{0};

 public:
  DoubleCompressedIter() = default;
  DoubleCompressedIter(CompressedByteT const *XGBOOST_RESTRICT buf0, std::size_t n0_bytes,
                       CompressedByteT const *XGBOOST_RESTRICT buf1, bst_idx_t n_symbols)
      : buf0_{buf0}, buf1_{buf1}, n0_{n0_bytes}, symbol_bits_{detail::SymbolBits(n_symbols)} {}

  XGBOOST_DEVICE reference operator[](std::size_t idx) const {
    constexpr std::int32_t kBitsPerByte = 8;

    std::size_t start_bit_idx = ((idx + 1) * symbol_bits_ - 1);
    std::size_t start_byte_idx = start_bit_idx / kBitsPerByte;
    start_byte_idx += detail::kPadding;

    std::uint64_t tmp;

    if (start_byte_idx >= this->n0_ && (start_byte_idx - 4) < this->n0_) {
      // Access between two buffers.
      auto getv = [&](auto shift) {
        auto shifted = start_byte_idx - shift;
        bool ind = (shifted >= n0_);  // indicator for which buffer to read
        // Pick the buffer to read
        auto const *XGBOOST_RESTRICT buf = ind ? buf1_ : buf0_;
        shifted -= ind * n0_;
        return static_cast<std::uint64_t>(buf[shifted]);
      };
      // Read 5 bytes - the maximum we will need
      tmp = static_cast<std::uint64_t>(buf0_[start_byte_idx - 4]) << 32 | getv(3) << 24 |
            getv(2) << 16 | getv(1) << 8 | static_cast<std::uint64_t>(buf1_[start_byte_idx - n0_]);
    } else {
      // Access one of the buffers
      bool ind = start_byte_idx >= n0_;
      // Pick the buffer to read
      auto const *XGBOOST_RESTRICT buf = reinterpret_cast<CompressedByteT const *>(
          (!ind) * reinterpret_cast<std::uintptr_t>(buf0_) +
          ind * reinterpret_cast<std::uintptr_t>(buf1_));
      // shifted start_byte_idx for buffer-local indexing.
      auto shifted = start_byte_idx - n0_ * ind;

      // Read 5 bytes - the maximum we will need

      // We don't have vector load here as we might create out-of-bound access due to down
      // alignment for the second buffer.
      tmp = static_cast<std::uint64_t>(buf[shifted - 4]) << 32 |
            static_cast<std::uint64_t>(buf[shifted - 3]) << 24 |
            static_cast<std::uint64_t>(buf[shifted - 2]) << 16 |
            static_cast<std::uint64_t>(buf[shifted - 1]) << 8 | buf[shifted];
    }

    // Knock out the unneeded bits from the right
    std::int32_t bit_shift = (kBitsPerByte - ((idx + 1) * symbol_bits_)) % kBitsPerByte;
    tmp >>= bit_shift;
    // Take exactly symbol_bits_ number of bits by masking off unneeded bits.
    std::uint64_t mask = (static_cast<std::uint64_t>(1) << symbol_bits_) - 1;
    return static_cast<OutT>(tmp & mask);
  }
};
}  // namespace xgboost::common
