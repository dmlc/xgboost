/**
 * Copyright 2019-2023, by XGBoost Contributors
 */
#if !defined(NOMINMAX) && defined(_WIN32)
#define NOMINMAX
#endif  // !defined(NOMINMAX)

#if !defined(xgboost_IS_WIN)

#if defined(_MSC_VER) || defined(__MINGW32__)
#define xgboost_IS_WIN 1
#endif  // defined(_MSC_VER) || defined(__MINGW32__)

#endif  // !defined(xgboost_IS_WIN)

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>     // for open, O_RDONLY
#include <sys/mman.h>  // for mmap, mmap64, munmap
#include <unistd.h>    // for close, getpagesize
#elif defined(xgboost_IS_WIN)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif  // defined(__unix__)

#include <algorithm>     // for copy, transform
#include <cctype>        // for tolower
#include <cerrno>        // for errno
#include <cstddef>       // for size_t
#include <cstdint>       // for int32_t, uint32_t
#include <cstring>       // for memcpy
#include <fstream>       // for ifstream
#include <iterator>      // for distance
#include <limits>        // for numeric_limits
#include <memory>        // for unique_ptr
#include <string>        // for string
#include <system_error>  // for error_code, system_category
#include <utility>       // for move
#include <vector>        // for vector

#include "io.h"
#include "xgboost/collective/socket.h"  // for LastError
#include "xgboost/logging.h"

namespace xgboost::common {
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

namespace {
// Get system alignment value for IO with mmap.
std::size_t GetMmapAlignment() {
#if defined(xgboost_IS_WIN)
  SYSTEM_INFO sys_info;
  GetSystemInfo(&sys_info);
  // During testing, `sys_info.dwPageSize` is of size 4096 while `dwAllocationGranularity` is of
  // size 65536.
  return sys_info.dwAllocationGranularity;
#else
  return getpagesize();
#endif
}

auto SystemErrorMsg() {
  std::int32_t errsv = system::LastError();
  auto err = std::error_code{errsv, std::system_category()};
  return err.message();
}
}  // anonymous namespace

std::string LoadSequentialFile(std::string uri, bool stream) {
  auto OpenErr = [&uri]() {
    std::string msg;
    msg = "Opening " + uri + " failed: ";
    msg += SystemErrorMsg();
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

struct PrivateMmapConstStream::MMAPFile {
#if defined(xgboost_IS_WIN)
  HANDLE fd{INVALID_HANDLE_VALUE};
  HANDLE file_map{INVALID_HANDLE_VALUE};
#else
  std::int32_t fd{0};
#endif
  char* base_ptr{nullptr};
  std::size_t base_size{0};
  std::string path;
};

char* PrivateMmapConstStream::Open(std::string path, std::size_t offset, std::size_t length) {
  if (length == 0) {
    return nullptr;
  }

#if defined(xgboost_IS_WIN)
  HANDLE fd = CreateFile(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                         FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);
  CHECK_NE(fd, INVALID_HANDLE_VALUE) << "Failed to open:" << path << ". " << SystemErrorMsg();
#else
  auto fd = open(path.c_str(), O_RDONLY);
  CHECK_GE(fd, 0) << "Failed to open:" << path << ". " << SystemErrorMsg();
#endif

  char* ptr{nullptr};
  // Round down for alignment.
  auto view_start = offset / GetMmapAlignment() * GetMmapAlignment();
  auto view_size = length + (offset - view_start);

#if defined(__linux__) || defined(__GLIBC__)
  int prot{PROT_READ};
  ptr = reinterpret_cast<char*>(mmap64(nullptr, view_size, prot, MAP_PRIVATE, fd, view_start));
  CHECK_NE(ptr, MAP_FAILED) << "Failed to map: " << path << ". " << SystemErrorMsg();
  handle_.reset(new MMAPFile{fd, ptr, view_size, std::move(path)});
#elif defined(xgboost_IS_WIN)
  auto file_size = GetFileSize(fd, nullptr);
  DWORD access = PAGE_READONLY;
  auto map_file = CreateFileMapping(fd, nullptr, access, 0, file_size, nullptr);
  access = FILE_MAP_READ;
  std::uint32_t loff = static_cast<std::uint32_t>(view_start);
  std::uint32_t hoff = view_start >> 32;
  CHECK(map_file) << "Failed to map: " << path << ". " << SystemErrorMsg();
  ptr = reinterpret_cast<char*>(MapViewOfFile(map_file, access, hoff, loff, view_size));
  CHECK_NE(ptr, nullptr) << "Failed to map: " << path << ". " << SystemErrorMsg();
  handle_.reset(new MMAPFile{fd, map_file, ptr, view_size, std::move(path)});
#else
  CHECK_LE(offset, std::numeric_limits<off_t>::max())
      << "File size has exceeded the limit on the current system.";
  int prot{PROT_READ};
  ptr = reinterpret_cast<char*>(mmap(nullptr, view_size, prot, MAP_PRIVATE, fd, view_start));
  CHECK_NE(ptr, MAP_FAILED) << "Failed to map: " << path << ". " << SystemErrorMsg();
  handle_.reset(new MMAPFile{fd, ptr, view_size, std::move(path)});
#endif  // defined(__linux__)

  ptr += (offset - view_start);
  return ptr;
}

PrivateMmapConstStream::PrivateMmapConstStream(std::string path, std::size_t offset,
                                               std::size_t length)
    : MemoryFixSizeBuffer{}, handle_{nullptr} {
  this->p_buffer_ = Open(std::move(path), offset, length);
  this->buffer_size_ = length;
}

PrivateMmapConstStream::~PrivateMmapConstStream() {
  CHECK(handle_);
#if defined(xgboost_IS_WIN)
  if (p_buffer_) {
    CHECK(UnmapViewOfFile(handle_->base_ptr)) "Faled to call munmap: " << SystemErrorMsg();
  }
  if (handle_->fd != INVALID_HANDLE_VALUE) {
    CHECK(CloseHandle(handle_->fd)) << "Failed to close handle: " << SystemErrorMsg();
  }
  if (handle_->file_map != INVALID_HANDLE_VALUE) {
    CHECK(CloseHandle(handle_->file_map)) << "Failed to close mapping object: " << SystemErrorMsg();
  }
#else
  if (handle_->base_ptr) {
    CHECK_NE(munmap(handle_->base_ptr, handle_->base_size), -1)
        << "Faled to call munmap: " << handle_->path << ". " << SystemErrorMsg();
  }
  if (handle_->fd != 0) {
    CHECK_NE(close(handle_->fd), -1)
        << "Faled to close: " << handle_->path << ". " << SystemErrorMsg();
  }
#endif
}
}  // namespace xgboost::common

#if defined(xgboost_IS_WIN)
#undef xgboost_IS_WIN
#endif  // defined(xgboost_IS_WIN)
