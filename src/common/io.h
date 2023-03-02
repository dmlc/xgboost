/*!
 * Copyright by XGBoost Contributors 2014-2022
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
#include <fstream>

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
  explicit PeekableInStream(dmlc::Stream* strm) : strm_(strm) {}

  size_t Read(void* dptr, size_t size) override;
  virtual size_t PeekRead(void* dptr, size_t size);

  void Write(const void*, size_t) override {
    LOG(FATAL) << "Not implemented";
  }

 private:
  /*! \brief input stream */
  dmlc::Stream *strm_;
  /*! \brief current buffer pointer */
  size_t buffer_ptr_{0};
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
  ~FixedSizeStream() override = default;

  size_t Read(void* dptr, size_t size) override;
  size_t PeekRead(void* dptr, size_t size) override;
  size_t Size() const { return buffer_.size(); }
  size_t Tell() const { return pointer_; }
  void Seek(size_t pos);

  void Write(const void*, size_t) override {
    LOG(FATAL) << "Not implemented";
  }

  /*!
   *  \brief Take the buffer from `FixedSizeStream'.  The one in `FixedSizeStream' will be
   *  cleared out.
   */
  void Take(std::string* out);

 private:
  size_t pointer_{0};
  std::string buffer_;
};

/*!
 * \brief Helper function for loading consecutive file to avoid dmlc Stream when possible.
 *
 * \param uri    URI or file name to file.
 * \param stream Use dmlc Stream unconditionally if set to true.  Used for running test
 *               without remote filesystem.
 *
 * \return File content.
 */
std::string LoadSequentialFile(std::string uri, bool stream = false);

/**
 * \brief Get file extension from file name.
 *
 * \param  lower Return in lower case.
 *
 * \return File extension without the `.`
 */
std::string FileExtension(std::string fname, bool lower = true);

/**
 * \brief Read the whole buffer from dmlc stream.
 */
inline std::string ReadAll(dmlc::Stream* fi, PeekableInStream* fp) {
  std::string buffer;
  if (auto fixed_size = dynamic_cast<common::MemoryFixSizeBuffer*>(fi)) {
    fixed_size->Seek(common::MemoryFixSizeBuffer::kSeekEnd);
    size_t size = fixed_size->Tell();
    buffer.resize(size);
    fixed_size->Seek(0);
    CHECK_EQ(fixed_size->Read(&buffer[0], size), size);
  } else {
    FixedSizeStream{fp}.Take(&buffer);
  }
  return buffer;
}

/**
 * \brief Read the whole file content into a string.
 */
inline std::string ReadAll(std::string const &path) {
  std::ifstream stream(path);
  if (!stream.is_open()) {
    LOG(FATAL) << "Could not open file " << path;
  }
  std::string content{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
  if (content.empty()) {
    LOG(FATAL) << "Empty file " << path;
  }
  return content;
}

}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_IO_H_
