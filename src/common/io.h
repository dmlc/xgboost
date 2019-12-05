/*!
 * Copyright 2014 by Contributors
 * \file io.h
 * \brief general stream interface for serialization, I/O
 * \author Tianqi Chen
 */

#ifndef XGBOOST_COMMON_IO_H_
#define XGBOOST_COMMON_IO_H_

#include <dmlc/io.h>
#include <rabit/rabit.h>
#include <string>
#include <cstring>

#include "common.h"

namespace xgboost {
namespace common {
using MemoryFixSizeBuffer = rabit::utils::MemoryFixSizeBuffer;
using MemoryBufferStream = rabit::utils::MemoryBufferStream;

/*!
 * \brief Input stream that support additional PeekRead operation,
 *  besides read.
 */
class PeekableInStream : public dmlc::Stream {
 public:
  explicit PeekableInStream(dmlc::Stream* strm)
      : strm_(strm), buffer_ptr_(0) {}

  size_t Read(void* dptr, size_t size) override;
  virtual size_t PeekRead(void* dptr, size_t size);

  void Write(const void* dptr, size_t size) override {
    LOG(FATAL) << "Not implemented";
  }

 private:
  /*! \brief input stream */
  dmlc::Stream *strm_;
  /*! \brief current buffer pointer */
  size_t buffer_ptr_;
  /*! \brief internal buffer */
  std::string buffer_;
};
/*!
 * \brief A simple class used to consume `dmlc::Stream' all at once.
 *
 * With it one can load the rabit checkpoint into a known size string buffer.
 */
class FixedSizeStream : public PeekableInStream {
 public:
  explicit FixedSizeStream(PeekableInStream* stream);
  ~FixedSizeStream() = default;

  size_t Read(void* dptr, size_t size) override;
  size_t PeekRead(void* dptr, size_t size) override;
  size_t Size() const { return buffer_.size(); }
  size_t Tell() const { return pointer_; }
  void Seek(size_t pos);

  void Write(const void* dptr, size_t size) override {
    LOG(FATAL) << "Not implemented";
  }

  /*!
   *  \brief Take the buffer from `FixedSizeStream'.  The one in `FixedSizeStream' will be
   *  cleared out.
   */
  void Take(std::string* out);

 private:
  size_t pointer_;
  std::string buffer_;
};

// Optimized for consecutive file loading in unix like systime.
std::string LoadSequentialFile(std::string fname);

inline std::string FileExtension(std::string const& fname) {
  auto splited = Split(fname, '.');
  if (splited.size() > 1) {
    return splited.back();
  } else {
    return "";
  }
}

}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_IO_H_
