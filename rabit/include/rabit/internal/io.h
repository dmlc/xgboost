/*!
 *  Copyright (c) 2014-2019 by Contributors
 * \file io.h
 * \brief utilities with different serializable implementations
 * \author Tianqi Chen
 */
#ifndef RABIT_INTERNAL_IO_H_
#define RABIT_INTERNAL_IO_H_
#include <cstdio>
#include <vector>
#include <cstring>
#include <string>
#include <algorithm>
#include <numeric>
#include <limits>
#include "rabit/internal/utils.h"
#include "rabit/serializable.h"

namespace rabit {
namespace utils {
/*! \brief re-use definition of dmlc::SeekStream */
using SeekStream = dmlc::SeekStream;
/*! \brief fixed size memory buffer */
struct MemoryFixSizeBuffer : public SeekStream {
 public:
  // similar to SEEK_END in libc
  static size_t constexpr kSeekEnd = std::numeric_limits<size_t>::max();

 public:
  MemoryFixSizeBuffer(void *p_buffer, size_t buffer_size)
      : p_buffer_(reinterpret_cast<char*>(p_buffer)),
        buffer_size_(buffer_size) {
    curr_ptr_ = 0;
  }
  ~MemoryFixSizeBuffer() override = default;
  size_t Read(void *ptr, size_t size) override {
    size_t nread = std::min(buffer_size_ - curr_ptr_, size);
    if (nread != 0) std::memcpy(ptr, p_buffer_ + curr_ptr_, nread);
    curr_ptr_ += nread;
    return nread;
  }
  void Write(const void *ptr, size_t size) override {
    if (size == 0) return;
    utils::Assert(curr_ptr_ + size <=  buffer_size_,
                  "write position exceed fixed buffer size");
    std::memcpy(p_buffer_ + curr_ptr_, ptr, size);
    curr_ptr_ += size;
  }
  void Seek(size_t pos) override {
    if (pos == kSeekEnd) {
      curr_ptr_ = buffer_size_;
    } else {
      curr_ptr_ = static_cast<size_t>(pos);
    }
  }
  size_t Tell() override {
    return curr_ptr_;
  }
  virtual bool AtEnd() const {
    return curr_ptr_ == buffer_size_;
  }

 private:
  /*! \brief in memory buffer */
  char *p_buffer_;
  /*! \brief current pointer */
  size_t buffer_size_;
  /*! \brief current pointer */
  size_t curr_ptr_;
};  // class MemoryFixSizeBuffer

/*! \brief a in memory buffer that can be read and write as stream interface */
struct MemoryBufferStream : public SeekStream {
 public:
  explicit MemoryBufferStream(std::string *p_buffer)
      : p_buffer_(p_buffer) {
    curr_ptr_ = 0;
  }
  ~MemoryBufferStream() override = default;
  size_t Read(void *ptr, size_t size) override {
    utils::Assert(curr_ptr_ <= p_buffer_->length(),
                  "read can not have position excceed buffer length");
    size_t nread = std::min(p_buffer_->length() - curr_ptr_, size);
    if (nread != 0) std::memcpy(ptr, &(*p_buffer_)[0] + curr_ptr_, nread);
    curr_ptr_ += nread;
    return nread;
  }
  void Write(const void *ptr, size_t size) override {
    if (size == 0) return;
    if (curr_ptr_ + size > p_buffer_->length()) {
      p_buffer_->resize(curr_ptr_+size);
    }
    std::memcpy(&(*p_buffer_)[0] + curr_ptr_, ptr, size);
    curr_ptr_ += size;
  }
  void Seek(size_t pos) override {
    curr_ptr_ = static_cast<size_t>(pos);
  }
  size_t Tell() override {
    return curr_ptr_;
  }
  virtual bool AtEnd() const {
    return curr_ptr_ == p_buffer_->length();
  }

 private:
  /*! \brief in memory buffer */
  std::string *p_buffer_;
  /*! \brief current pointer */
  size_t curr_ptr_;
};  // class MemoryBufferStream
}  // namespace utils
}  // namespace rabit
#endif  // RABIT_INTERNAL_IO_H_
