#ifndef RABIT_UTILS_IO_H
#define RABIT_UTILS_IO_H
#include <cstdio>
#include <vector>
#include <cstring>
#include <string>
#include "./utils.h"
#include "../rabit_serializable.h"
/*!
 * \file io.h
 * \brief utilities that implements different serializable interface
 * \author Tianqi Chen
 */
namespace rabit {
namespace utils {
/*! \brief interface of i/o stream that support seek */
class ISeekStream: public IStream {
 public:
  /*! \brief seek to certain position of the file */
  virtual void Seek(size_t pos) = 0;
  /*! \brief tell the position of the stream */
  virtual size_t Tell(void) = 0;
};

/*! \brief fixed size memory buffer */
struct MemoryFixSizeBuffer : public ISeekStream {
 public:
  MemoryFixSizeBuffer(void *p_buffer, size_t buffer_size) 
      : p_buffer_(reinterpret_cast<char*>(p_buffer)), buffer_size_(buffer_size) {
    curr_ptr_ = 0;
  }
  virtual ~MemoryFixSizeBuffer(void) {}
  virtual size_t Read(void *ptr, size_t size) {
    utils::Assert(curr_ptr_ + size <= buffer_size_,
                  "read can not have position excceed buffer length");
    size_t nread = std::min(buffer_size_ - curr_ptr_, size);
    if (nread != 0) memcpy(ptr, p_buffer_ + curr_ptr_, nread);
    curr_ptr_ += nread;
    return nread;
  }
  virtual void Write(const void *ptr, size_t size) {
    if (size == 0) return;
    utils::Assert(curr_ptr_ + size <=  buffer_size_, 
                  "write position exceed fixed buffer size");
    memcpy(p_buffer_ + curr_ptr_, ptr, size);
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
}; // class MemoryFixSizeBuffer

/*! \brief a in memory buffer that can be read and write as stream interface */
struct MemoryBufferStream : public ISeekStream {
 public:
  MemoryBufferStream(std::string *p_buffer) 
      : p_buffer_(p_buffer) {
    curr_ptr_ = 0;
  }
  virtual ~MemoryBufferStream(void) {}
  virtual size_t Read(void *ptr, size_t size) {
    utils::Assert(curr_ptr_ <= p_buffer_->length(),
                  "read can not have position excceed buffer length");
    size_t nread = std::min(p_buffer_->length() - curr_ptr_, size);
    if (nread != 0) memcpy(ptr, &(*p_buffer_)[0] + curr_ptr_, nread);
    curr_ptr_ += nread;
    return nread;
  }
  virtual void Write(const void *ptr, size_t size) {
    if (size == 0) return;
    if (curr_ptr_ + size > p_buffer_->length()) {
      p_buffer_->resize(curr_ptr_+size);
    }
    memcpy(&(*p_buffer_)[0] + curr_ptr_, ptr, size); 
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
}; // class MemoryBufferStream

/*! \brief implementation of file i/o stream */
class FileStream : public ISeekStream {
 public:
  explicit FileStream(FILE *fp) : fp(fp) {}
  explicit FileStream(void) {
    this->fp = NULL;
  }
  virtual size_t Read(void *ptr, size_t size) {
    return std::fread(ptr, size, 1, fp);
  }
  virtual void Write(const void *ptr, size_t size) {
    std::fwrite(ptr, size, 1, fp);
  }
  virtual void Seek(size_t pos) {
    std::fseek(fp, static_cast<long>(pos), SEEK_SET);
  }
  virtual size_t Tell(void) {
    return std::ftell(fp);
  }
  inline void Close(void) {
    if (fp != NULL){
      std::fclose(fp); fp = NULL;
    }
  }

 private:
  FILE *fp;
};
}  // namespace utils
}  // namespace rabit
#endif
