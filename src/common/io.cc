/*!
 * Copyright (c) by XGBoost Contributors 2019
 */
#if defined(__unix__)
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif  // defined(__unix__)
#include <algorithm>
#include <cstdio>
#include <string>
#include <utility>

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

FixedSizeStream::FixedSizeStream(PeekableInStream* stream) : PeekableInStream(stream), pointer_{0} {
  size_t constexpr kInitialSize = 4096;
  size_t size {kInitialSize}, total {0};
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

std::string LoadSequentialFile(std::string fname) {
  auto OpenErr = [&fname]() {
                   std::string msg;
                   msg = "Opening " + fname + " failed: ";
                   msg += strerror(errno);
                   LOG(FATAL) << msg;
                 };
  auto ReadErr = [&fname]() {
                   std::string msg {"Error in reading file: "};
                   msg += fname;
                   msg += ": ";
                   msg += strerror(errno);
                   LOG(FATAL) << msg;
                 };

  std::string buffer;
#if defined(__unix__)
  struct stat fs;
  if (stat(fname.c_str(), &fs) != 0) {
    OpenErr();
  }

  size_t f_size_bytes = fs.st_size;
  buffer.resize(f_size_bytes + 1);
  int32_t fd = open(fname.c_str(), O_RDONLY);
#if defined(__linux__)
  posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif  // defined(__linux__)
  ssize_t bytes_read = read(fd, &buffer[0], f_size_bytes);
  if (bytes_read < 0) {
    close(fd);
    ReadErr();
  }
  close(fd);
#else  // defined(__unix__)
  FILE *f = fopen(fname.c_str(), "r");
  if (f == NULL) {
    std::string msg;
    OpenErr();
  }
  fseek(f, 0, SEEK_END);
  auto fsize = ftell(f);
  fseek(f, 0, SEEK_SET);

  buffer.resize(fsize + 1);
  fread(&buffer[0], 1, fsize, f);
  fclose(f);
#endif  // defined(__unix__)
  buffer.back() = '\0';
  return buffer;
}

}  // namespace common
}  // namespace xgboost
