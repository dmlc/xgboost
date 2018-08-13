/*!
 * Copyright 2017 by Contributors
 * \file compressed_iterator.h
 */
#pragma once
#include <xgboost/base.h>
#include <cmath>
#include <cstddef>
#include <algorithm>

#ifdef __CUDACC__
#include "device_helpers.cuh"
#endif

namespace xgboost {
namespace common {

using CompressedByteT = unsigned char;

namespace detail {
inline void SetBit(CompressedByteT *byte, int bit_idx) {
  *byte |= 1 << bit_idx;
}
template <typename T>
inline T CheckBit(const T &byte, int bit_idx) {
  return byte & (1 << bit_idx);
}
inline void ClearBit(CompressedByteT *byte, int bit_idx) {
  *byte &= ~(1 << bit_idx);
}
static const int kPadding = 4;  // Assign padding so we can read slightly off
                               // the beginning of the array

// The number of bits required to represent a given unsigned range
static size_t SymbolBits(size_t num_symbols) {
  auto bits = std::ceil(std::log2(num_symbols));
  return std::max(static_cast<size_t>(bits), size_t(1));
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
 private:
  size_t symbol_bits_;
  size_t offset_;

 public:
  explicit CompressedBufferWriter(size_t num_symbols) : offset_(0) {
    symbol_bits_ = detail::SymbolBits(num_symbols);
  }

  /**
   * \fn  static size_t CompressedBufferWriter::CalculateBufferSize(int
   * num_elements, int num_symbols)
   *
   * \brief Calculates number of bytes requiredm for a given number of elements
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
    const int bits_per_byte = 8;
    size_t compressed_size = static_cast<size_t>(std::ceil(
        static_cast<double>(detail::SymbolBits(num_symbols) * num_elements) /
        bits_per_byte));
    return compressed_size + detail::kPadding;
  }

  template <typename T>
  void WriteSymbol(CompressedByteT *buffer, T symbol, size_t offset) {
    const int bits_per_byte = 8;

    for (size_t i = 0; i < symbol_bits_; i++) {
      size_t byte_idx = ((offset + 1) * symbol_bits_ - (i + 1)) / bits_per_byte;
      byte_idx += detail::kPadding;
      size_t bit_idx =
          ((bits_per_byte + i) - ((offset + 1) * symbol_bits_)) % bits_per_byte;

      if (detail::CheckBit(symbol, i)) {
        detail::SetBit(&buffer[byte_idx], bit_idx);
      } else {
        detail::ClearBit(&buffer[byte_idx], bit_idx);
      }
    }
  }

#ifdef __CUDACC__
  __device__ void AtomicWriteSymbol
    (CompressedByteT* buffer, uint64_t symbol, size_t offset) {
    size_t ibit_start = offset * symbol_bits_;
    size_t ibit_end = (offset + 1) * symbol_bits_ - 1;
    size_t ibyte_start = ibit_start / 8, ibyte_end = ibit_end / 8;

    symbol <<= 7 - ibit_end % 8;
    for (ptrdiff_t ibyte = ibyte_end; ibyte >= (ptrdiff_t)ibyte_start; --ibyte) {
      dh::AtomicOrByte(reinterpret_cast<unsigned int*>(buffer + detail::kPadding),
                       ibyte, symbol & 0xff);
      symbol >>= 8;
    }
  }
#endif

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
          buffer[buffer_position] = static_cast<CompressedByteT>(
              tmp >> (stored_bits - (j + 1) * 8));
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
    int tmp_bytes =
        static_cast<int>(std::ceil(static_cast<float>(stored_bits) / 8));
    for (int j = 0; j < tmp_bytes; j++) {
      int shift_bits = static_cast<int>(stored_bits) - (j + 1) * 8;
      if (shift_bits >= 0) {
        buffer[buffer_position] =
            static_cast<CompressedByteT>(tmp >> shift_bits);
      } else {
        buffer[buffer_position] =
            static_cast<CompressedByteT>(tmp << std::abs(shift_bits));
      }
      buffer_position++;
    }
  }
};

template <typename T>

/**
 * \class CompressedIterator
 *
 * \brief Read symbols from a bit compressed memory buffer. Usable on device and
 * host.
 *
 * \author  Rory
 * \date  7/9/2017
 */

class CompressedIterator {
 public:
  // Type definitions for thrust
  typedef CompressedIterator<T> self_type;  // NOLINT
  typedef ptrdiff_t difference_type;        // NOLINT
  typedef T value_type;                     // NOLINT
  typedef value_type *pointer;              // NOLINT
  typedef value_type reference;             // NOLINT

 private:
  CompressedByteT *buffer_;
  size_t symbol_bits_;
  size_t offset_;

 public:
  CompressedIterator() : buffer_(nullptr), symbol_bits_(0), offset_(0) {}
  CompressedIterator(CompressedByteT *buffer, int num_symbols)
      : buffer_(buffer), offset_(0) {
    symbol_bits_ = detail::SymbolBits(num_symbols);
  }

  XGBOOST_DEVICE reference operator*() const {
    const int bits_per_byte = 8;
    size_t start_bit_idx = ((offset_ + 1) * symbol_bits_ - 1);
    size_t start_byte_idx = start_bit_idx / bits_per_byte;
    start_byte_idx += detail::kPadding;

    // Read 5 bytes - the maximum we will need
    uint64_t tmp = static_cast<uint64_t>(buffer_[start_byte_idx - 4]) << 32 |
                   static_cast<uint64_t>(buffer_[start_byte_idx - 3]) << 24 |
                   static_cast<uint64_t>(buffer_[start_byte_idx - 2]) << 16 |
                   static_cast<uint64_t>(buffer_[start_byte_idx - 1]) << 8 |
                   buffer_[start_byte_idx];
    int bit_shift =
        (bits_per_byte - ((offset_ + 1) * symbol_bits_)) % bits_per_byte;
    tmp >>= bit_shift;
    // Mask off unneeded bits
    uint64_t mask = (1 << symbol_bits_) - 1;
    return static_cast<T>(tmp & mask);
  }

  XGBOOST_DEVICE reference operator[](size_t idx) const {
    self_type offset = (*this);
    offset.offset_ += idx;
    return *offset;
  }
};
}  // namespace common
}  // namespace xgboost
