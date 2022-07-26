/*!
 * Copyright 2017 by Contributors
 * \file compressed_iterator.h
 */
#pragma once
#include <xgboost/base.h>
#include <cmath>
#include <cstddef>
#include <algorithm>

#include "common.h"

#ifdef __CUDACC__
#include "device_helpers.cuh"
#endif  // __CUDACC__

namespace xgboost {
namespace common {

using CompressedByteT = unsigned char;

namespace detail {
// The number of bits required to represent a given unsigned range
inline XGBOOST_DEVICE size_t SymbolBits(size_t num_symbols) {
  auto bits = std::ceil(log2(static_cast<double>(num_symbols)));
  return common::Max(static_cast<size_t>(bits), size_t(1));
}
}  // namespace detail

inline XGBOOST_DEVICE int SmallestWordSize(size_t num_symbols) {
  int word_size = 32;
  int bits = detail::SymbolBits(num_symbols);
  if (bits <= 16) {
    word_size = 16;
  }
  if (bits <= 8) {
    word_size = 8;
  }
  return word_size;
}

class CompressedWriter {
  CompressedByteT *buffer_ {nullptr};
  int symbol_bits_ {0};

 public:
  CompressedWriter(CompressedByteT *buffer, size_t num_symbols) : buffer_(buffer) {
    symbol_bits_ = SmallestWordSize(num_symbols);
  }

  static size_t CalculateBufferSize(size_t num_elements, size_t num_symbols) {
    return (SmallestWordSize(num_symbols)/8)*num_elements;
  }

  XGBOOST_DEVICE void Write(size_t idx, uint32_t x) {
    if (symbol_bits_ == 8) {
      buffer_[idx] = x;
    } else if (symbol_bits_ == 16) {
      reinterpret_cast<uint16_t *>(buffer_)[idx] = x;
    } else if (symbol_bits_ == 32) {
      reinterpret_cast<uint32_t *>(buffer_)[idx] = x;
    }
  }
};

class CompressedIterator {
 public:
  // Type definitions for thrust
  typedef CompressedIterator self_type;  // NOLINT
  typedef ptrdiff_t difference_type;        // NOLINT
  typedef uint32_t value_type;                     // NOLINT
  typedef value_type *pointer;              // NOLINT
  typedef value_type& reference;             // NOLINT

 private:
  const CompressedByteT *buffer_ {nullptr};
  int symbol_bits_ {0};

 public:
  CompressedIterator() = default;
  CompressedIterator(const CompressedByteT *buffer, size_t num_symbols)
      : buffer_(buffer) {
    symbol_bits_ = SmallestWordSize(num_symbols);
  }
  XGBOOST_DEVICE value_type operator[](size_t idx) const {
    if (symbol_bits_ == 8) {
      return buffer_[idx];
    } else if (symbol_bits_ == 16) {
      return reinterpret_cast<const uint16_t *>(buffer_)[idx];
    } else {
      return reinterpret_cast<const uint32_t *>(buffer_)[idx];
    }
  }
};
}  // namespace common
}  // namespace xgboost
