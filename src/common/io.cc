/**
 * Copyright 2019-2023, by XGBoost Contributors
 */
#if !defined(NOMINMAX) && defined(_WIN32)
#define NOMINMAX
#endif  // !defined(NOMINMAX)

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>     // for open, O_RDONLY
#include <sys/mman.h>  // for mmap, mmap64, munmap
#include <sys/stat.h>
#include <unistd.h>  // for close, getpagesize
#elif defined(_MSC_VER)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif               // defined(__unix__)

#include <algorithm>
#include <cerrno>  // for errno
#include <cstdio>
#include <fstream>
#include <limits>  // for numeric_limits
#include <memory>
#include <string>
#include <utility>
#include <vector>  // for vector

#include "io.h"
#include "xgboost/logging.h"
#include "xgboost/collective/socket.h"

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

std::size_t GetPageSize() {
#if defined(_MSC_VER)
  SYSTEM_INFO sys_info;
  GetSystemInfo(&sys_info);
  // During testing, `sys_info.dwPageSize` is of size 4096 while `dwAllocationGranularity` is of size 65536.
  return sys_info.dwAllocationGranularity;
#else
  return getpagesize();
#endif
}

std::size_t PadPageForMmap(std::size_t file_bytes, dmlc::Stream* fo) {
  decltype(file_bytes) page_size = GetPageSize();
  CHECK(page_size != 0 && page_size % 2 == 0) << "Failed to get page size on the current system.";
  CHECK_NE(file_bytes, 0) << "Empty page encountered.";
  auto n_pages = file_bytes / page_size + !!(file_bytes % page_size != 0);
  auto padded = n_pages * page_size;
  auto padding = padded - file_bytes;
  std::vector<std::uint8_t> padding_bytes(padding, 0);
  // fo->Write(padding_bytes.data(), padding_bytes.size());
  return file_bytes;
}

struct PrivateMmapStream::MMAPFile {
#if defined(_MSC_VER)
  HANDLE fd{ INVALID_HANDLE_VALUE };
#else
  std::int32_t fd {0};
#endif
  char* base_ptr{ nullptr };
  std::size_t base_size{0};
  std::string path;
};

PrivateMmapStream::PrivateMmapStream(std::string path, bool read_only, std::size_t offset,
                                     std::size_t length)
    : MemoryFixSizeBuffer{} {
  this->p_buffer_ = Open(std::move(path), read_only, offset, length);
  this->buffer_size_ = length;
}

char* PrivateMmapStream::Open(std::string path, bool read_only, std::size_t offset,
                              std::size_t length) {
#if defined(_MSC_VER)
  HANDLE fd = CreateFile(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                         FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);
  CHECK_NE(fd, INVALID_HANDLE_VALUE) << "Failed to open:" << path;
#else
  auto fd = open(path.c_str(), O_RDONLY);
  CHECK_GE(fd, 0) << "Failed to open:" << path << ". " << strerror(errno);
#endif

  char* ptr{nullptr};
  auto view_start = offset / GetPageSize() * GetPageSize();
  auto view_size = length + (offset - view_start);
  std::cout << view_start << " size: " << view_size << std::endl;
#if defined(__linux__) || defined(__GLIBC__)
  int prot{PROT_READ};
  if (!read_only) {
    prot |= PROT_WRITE;
  }
  ptr = reinterpret_cast<char*>(mmap64(nullptr, view_size, prot, MAP_PRIVATE, fd, view_start));
  CHECK_NE(ptr, MAP_FAILED) << "Failed to map: " << path << ". " << strerror(errno);
#elif defined(_MSC_VER)
  auto file_size = GetFileSize(fd, nullptr);
  DWORD access = read_only ? PAGE_READONLY : PAGE_READWRITE;
  auto map_file = CreateFileMapping(fd, nullptr, access, 0, file_size, nullptr);
  access = read_only ? FILE_MAP_READ : FILE_MAP_ALL_ACCESS;
  std::uint32_t loff = static_cast<std::uint32_t>(view_start);
  std::uint32_t hoff = view_start >> 32;
  CHECK(map_file) << "Failed to map: " << path << ". " << GetLastError();
  ptr = reinterpret_cast<char*>(MapViewOfFile(map_file, access, hoff, loff, view_size));
  if (ptr == nullptr) {
    system::ThrowAtError("MapViewOfFile");
  }
  CHECK_NE(ptr, nullptr) << "Failed to map: " << path << ". " << GetLastError();
#else
  CHECK_LE(offset, std::numeric_limits<off_t>::max())
      << "File size has exceeded the limit on the current system.";
  int prot{PROT_READ};
  if (!read_only) {
    prot |= PROT_WRITE;
  }
  ptr = reinterpret_cast<char*>(mmap(nullptr, view_size, prot, MAP_PRIVATE, fd, view_start));
  CHECK_NE(ptr, MAP_FAILED) << "Failed to map: " << path << ". " << strerror(errno);
#endif  // defined(__linux__)

  handle_.reset(new MMAPFile{ fd, ptr, view_size, std::move(path) });
  ptr += (offset - view_start);
  return ptr;
}

PrivateMmapStream::~PrivateMmapStream() {
  CHECK(handle_);
#if defined(_MSC_VER)
  if (p_buffer_) {
    CHECK(UnmapViewOfFile(handle_->base_ptr)) "Faled to munmap." << GetLastError();
  }
  if (handle_->fd != INVALID_HANDLE_VALUE) {
    CHECK(CloseHandle(handle_->fd));
  }
#else
  CHECK_NE(munmap(handle_->base_ptr, handle_->base_size), -1)
      << "Faled to munmap." << handle_->path << ". " << strerror(errno);
  CHECK_NE(close(handle_->fd), -1) << "Faled to close: " << handle_->path << ". " << strerror(errno);
#endif
}
}  // namespace common
}  // namespace xgboost
