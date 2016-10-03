/*!
 *  Copyright (c) 2015 by Contributors
 * \file memory_io.h
 * \brief defines binary serialization class to serialize things into/from memory region.
 */
#ifndef DMLC_MEMORY_IO_H_
#define DMLC_MEMORY_IO_H_

#include <cstring>
#include <string>
#include <algorithm>
#include "./base.h"
#include "./io.h"
#include "./logging.h"

namespace dmlc {
/*!
 * \brief A Stream that operates on fixed region of memory
 *  This class allows us to read/write from/to a fixed memory region.
 */
struct MemoryFixedSizeStream : public SeekStream {
 public:
  /*!
   * \brief constructor
   * \param p_buffer the head pointer of the memory region.
   * \param buffer_size the size of the memorybuffer
   */
  MemoryFixedSizeStream(void *p_buffer, size_t buffer_size)
      : p_buffer_(reinterpret_cast<char*>(p_buffer)),
        buffer_size_(buffer_size) {
    curr_ptr_ = 0;
  }
  virtual size_t Read(void *ptr, size_t size) {
    CHECK(curr_ptr_ + size <= buffer_size_);
    size_t nread = std::min(buffer_size_ - curr_ptr_, size);
    if (nread != 0) std::memcpy(ptr, p_buffer_ + curr_ptr_, nread);
    curr_ptr_ += nread;
    return nread;
  }
  virtual void Write(const void *ptr, size_t size) {
    if (size == 0) return;
    CHECK(curr_ptr_ + size <=  buffer_size_);
    std::memcpy(p_buffer_ + curr_ptr_, ptr, size);
    curr_ptr_ += size;
  }
  virtual void Seek(size_t pos) {
    curr_ptr_ = static_cast<size_t>(pos);
  }
  virtual size_t Tell(void) {
    return curr_ptr_;
  }

 private:
  /*! \brief in memory buffer */
  char *p_buffer_;
  /*! \brief current pointer */
  size_t buffer_size_;
  /*! \brief current pointer */
  size_t curr_ptr_;
};  // class MemoryFixedSizeStream

/*!
 * \brief A in memory stream that is backed by std::string.
 *  This class allows us to read/write from/to a std::string.
 */
struct MemoryStringStream : public dmlc::SeekStream {
 public:
  /*!
   * \brief constructor
   * \param p_buffer the pointer to the string.
   */
  explicit MemoryStringStream(std::string *p_buffer)
      : p_buffer_(p_buffer) {
    curr_ptr_ = 0;
  }
  virtual size_t Read(void *ptr, size_t size) {
    CHECK(curr_ptr_ <= p_buffer_->length());
    size_t nread = std::min(p_buffer_->length() - curr_ptr_, size);
    if (nread != 0) std::memcpy(ptr, &(*p_buffer_)[0] + curr_ptr_, nread);
    curr_ptr_ += nread;
    return nread;
  }
  virtual void Write(const void *ptr, size_t size) {
    if (size == 0) return;
    if (curr_ptr_ + size > p_buffer_->length()) {
      p_buffer_->resize(curr_ptr_+size);
    }
    std::memcpy(&(*p_buffer_)[0] + curr_ptr_, ptr, size);
    curr_ptr_ += size;
  }
  virtual void Seek(size_t pos) {
    curr_ptr_ = static_cast<size_t>(pos);
  }
  virtual size_t Tell(void) {
    return curr_ptr_;
  }

 private:
  /*! \brief in memory buffer */
  std::string *p_buffer_;
  /*! \brief current pointer */
  size_t curr_ptr_;
};  // class MemoryStringStream
}  // namespace dmlc
#endif  // DMLC_MEMORY_IO_H_
