/*!
 * Copyright (c) by XGBoost Contributors 2019-2022
 */
#if defined(__unix__)
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif  // defined(__unix__)
#include <algorithm>
#include <fstream>
#include <string>
#include <memory>
#include <utility>
#include <cstdio>

#include "xgboost/logging.h"
#include "io.h"

namespace xgboost {
namespace common {

size_t PeekableInStream::Read(void* dptr, size_t size) {
  size_t nbuffer = buffer_.length() - buffer_ptr_;
  if (nbuffer == 0) return strm_->Read(dptr, size);
  if (nbuffer < size) {
    std::memcpy(dptr, dmlc::BeginPtr(buffer_) + buffer_ptr_, nbuffer);
    buffer_ptr_ += nbuffer;
    return nbuffer + strm_->Read(reinterpret_cast<char*>(dptr) + nbuffer,
                                 size - nbuffer);
  } else {
    std::memcpy(dptr, dmlc::BeginPtr(buffer_) + buffer_ptr_, size);
    buffer_ptr_ += size;
    return size;
  }
}

size_t PeekableInStream::PeekRead(void* dptr, size_t size) {
  size_t nbuffer = buffer_.length() - buffer_ptr_;
  if (nbuffer < size) {
    buffer_ = buffer_.substr(buffer_ptr_, buffer_.length());
    buffer_ptr_ = 0;
    buffer_.resize(size);
    size_t nadd = strm_->Read(dmlc::BeginPtr(buffer_) + nbuffer, size - nbuffer);
    buffer_.resize(nbuffer + nadd);
    std::memcpy(dptr, dmlc::BeginPtr(buffer_), buffer_.length());
    return buffer_.size();
  } else {
    std::memcpy(dptr, dmlc::BeginPtr(buffer_) + buffer_ptr_, size);
    return size;
  }
}

FixedSizeStream::FixedSizeStream(PeekableInStream* stream) : PeekableInStream(stream) {
  size_t constexpr kInitialSize = 4096;
  size_t size{kInitialSize}, total{0};
  buffer_.clear();
  while (true) {
    buffer_.resize(size);
    size_t read = stream->PeekRead(&buffer_[0], size);
    total = read;
    if (read < size) {
      break;
    }
    size *= 2;
  }
  buffer_.resize(total);
}

size_t FixedSizeStream::Read(void* dptr, size_t size) {
  auto read = this->PeekRead(dptr, size);
  pointer_ += read;
  return read;
}

size_t FixedSizeStream::PeekRead(void* dptr, size_t size) {
  if (size >= buffer_.size() - pointer_)  {
    std::copy(buffer_.cbegin() + pointer_, buffer_.cend(), reinterpret_cast<char*>(dptr));
    return std::distance(buffer_.cbegin() + pointer_, buffer_.cend());
  } else {
    auto const beg = buffer_.cbegin() + pointer_;
    auto const end = beg + size;
    std::copy(beg, end, reinterpret_cast<char*>(dptr));
    return std::distance(beg, end);
  }
}

void FixedSizeStream::Seek(size_t pos) {
  pointer_ = pos;
  CHECK_LE(pointer_, buffer_.size());
}

void FixedSizeStream::Take(std::string* out) {
  CHECK(out);
  *out = std::move(buffer_);
}

std::string LoadSequentialFile(std::string uri, bool stream) {
  auto OpenErr = [&uri]() {
    std::string msg;
    msg = "Opening " + uri + " failed: ";
    msg += strerror(errno);
    LOG(FATAL) << msg;
  };

  auto parsed = dmlc::io::URI(uri.c_str());
  // Read from file.
  if ((parsed.protocol == "file://" || parsed.protocol.length() == 0) && !stream) {
    std::string buffer;
    // Open in binary mode so that correct file size can be computed with
    // seekg(). This accommodates Windows platform:
    // https://docs.microsoft.com/en-us/cpp/standard-library/basic-istream-class?view=vs-2019#seekg
    std::ifstream ifs(uri, std::ios_base::binary | std::ios_base::in);
    if (!ifs) {
      // https://stackoverflow.com/a/17338934
      OpenErr();
    }

    ifs.seekg(0, std::ios_base::end);
    const size_t file_size = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0, std::ios_base::beg);
    buffer.resize(file_size + 1);
    ifs.read(&buffer[0], file_size);
    buffer.back() = '\0';

    return buffer;
  }

  // Read from remote.
  std::unique_ptr<dmlc::Stream> fs{dmlc::Stream::Create(uri.c_str(), "r")};
  std::string buffer;
  size_t constexpr kInitialSize = 4096;
  size_t size {kInitialSize}, total {0};
  while (true) {
    buffer.resize(total + size);
    size_t read = fs->Read(&buffer[total], size);
    total += read;
    if (read < size) {
      break;
    }
    size *= 2;
  }
  buffer.resize(total);
  return buffer;
}

std::string FileExtension(std::string fname, bool lower) {
  if (lower) {
    std::transform(fname.begin(), fname.end(), fname.begin(),
                   [](char c) { return std::tolower(c); });
  }
  auto splited = Split(fname, '.');
  if (splited.size() > 1) {
    return splited.back();
  } else {
    return "";
  }
}
}  // namespace common
}  // namespace xgboost
