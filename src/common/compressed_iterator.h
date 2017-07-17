/*!
 * Copyright 2017 by Contributors
 * \file compressed_iterator.h
 */
#pragma once
#include <xgboost/base.h>
#include <cmath>
#include <cstddef>
#include "dmlc/logging.h"

namespace xgboost {
namespace common {

typedef unsigned char compressed_byte_t;

namespace detail {
inline void SetBit(compressed_byte_t *byte, int bit_idx) {
  *byte |= 1 << bit_idx;
}
template <typename T>
inline T CheckBit(const T &byte, int bit_idx) {
  return byte & (1 << bit_idx);
}
inline void ClearBit(compressed_byte_t *byte, int bit_idx) {
  *byte &= ~(1 << bit_idx);
}
static const int padding = 4;  // Assign padding so we can read slightly off
                               // the beginning of the array

// The number of bits required to represent a given unsigned range
static int SymbolBits(int num_symbols) {
  return std::ceil(std::log2(num_symbols));
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
    size_t compressed_size = std::ceil(
        static_cast<double>(detail::SymbolBits(num_symbols) * num_elements) /
        bits_per_byte);
    return compressed_size + detail::padding;
  }

  template <typename T>
  void WriteSymbol(compressed_byte_t *buffer, T symbol, size_t offset) {
    const int bits_per_byte = 8;

    for (size_t i = 0; i < symbol_bits_; i++) {
      size_t byte_idx = ((offset + 1) * symbol_bits_ - (i + 1)) / bits_per_byte;
      byte_idx += detail::padding;
      size_t bit_idx =
          ((bits_per_byte + i) - ((offset + 1) * symbol_bits_)) % bits_per_byte;

      if (detail::CheckBit(symbol, i)) {
        detail::SetBit(&buffer[byte_idx], bit_idx);
      } else {
        detail::ClearBit(&buffer[byte_idx], bit_idx);
      }
    }
  }
  template <typename iter_t>
  void Write(compressed_byte_t *buffer, iter_t input_begin, iter_t input_end) {
    uint64_t tmp = 0;
    int stored_bits = 0;
    const int max_stored_bits = 64 - symbol_bits_;
    size_t buffer_position = detail::padding;
    const size_t num_symbols = input_end - input_begin;
    for (size_t i = 0; i < num_symbols; i++) {
      typename std::iterator_traits<iter_t>::value_type symbol = input_begin[i];
      if (stored_bits > max_stored_bits) {
        // Eject only full bytes
        size_t tmp_bytes = stored_bits / 8;
        for (size_t j = 0; j < tmp_bytes; j++) {
          buffer[buffer_position] = tmp >> (stored_bits - (j + 1) * 8);
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
    size_t tmp_bytes = std::ceil(static_cast<float>(stored_bits) / 8);
    for (size_t j = 0; j < tmp_bytes; j++) {
      int shift_bits = stored_bits - (j + 1) * 8;
      if (shift_bits >= 0) {
        buffer[buffer_position] = tmp >> shift_bits;
      } else {
        buffer[buffer_position] = tmp << std::abs(shift_bits);
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
  typedef CompressedIterator<T> self_type;  ///< My own type
  typedef ptrdiff_t
      difference_type;   ///< Type to express the result of subtracting
                         /// one iterator from another
  typedef T value_type;  ///< The type of the element the iterator can point to
  typedef value_type *pointer;   ///< The type of a pointer to an element the
                                 /// iterator can point to
  typedef value_type reference;  ///< The type of a reference to an element the
                                 /// iterator can point to
 private:
  compressed_byte_t *buffer_;
  size_t symbol_bits_;
  size_t offset_;

 public:
  CompressedIterator() : buffer_(nullptr), symbol_bits_(0), offset_(0) {}
  CompressedIterator(compressed_byte_t *buffer, int num_symbols)
      : buffer_(buffer), offset_(0) {
    symbol_bits_ = detail::SymbolBits(num_symbols);
  }

  XGBOOST_DEVICE reference operator*() const {
    const int bits_per_byte = 8;
    size_t start_bit_idx = ((offset_ + 1) * symbol_bits_ - 1);
    size_t start_byte_idx = start_bit_idx / bits_per_byte;
    start_byte_idx += detail::padding;

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
