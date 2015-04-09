#ifndef RABIT_LEARN_IO_BUFFER_READER_INL_H_
#define RABIT_LEARN_IO_BUFFER_READER_INL_H_
/*!
 * \file buffer_reader-inl.h
 * \brief implementation of stream buffer reader
 * \author Tianqi Chen
 */
#include "./io.h"

namespace rabit {
namespace io {
/*! \brief buffer reader of the stream that allows you to get */
class StreamBufferReader {
 public:
  StreamBufferReader(size_t buffer_size)
      :stream_(NULL),
       read_len_(1), read_ptr_(1) {
    buffer_.resize(buffer_size);
  }
  /*!
   * \brief set input stream
   */
  inline void set_stream(Stream *stream) {
    stream_ = stream;
    read_len_ = read_ptr_ = 1;
  }
  /*!
   * \brief allows quick read using get char
   */
  inline char GetChar(void) {
    while (true) {
      if (read_ptr_ < read_len_) {
        return buffer_[read_ptr_++];
      } else {
        read_len_ = stream_->Read(&buffer_[0], buffer_.length());
        if (read_len_ == 0) return EOF;
        read_ptr_ = 0;
      }
    }
  }
  /*! \brief whether we are reaching the end of file */
  inline bool AtEnd(void) const {
    return read_len_ == 0;
  }
  
 private:
  /*! \brief the underlying stream */
  Stream *stream_;
  /*! \brief buffer to hold data */
  std::string buffer_;
  /*! \brief length of valid data in buffer */
  size_t read_len_;
  /*! \brief pointer in the buffer */
  size_t read_ptr_;
};
}  // namespace io
}  // namespace rabit
#endif  // RABIT_LEARN_IO_BUFFER_READER_INL_H_
