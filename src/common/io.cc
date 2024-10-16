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
#include <sys/mman.h>  // for mmap, munmap, madvise
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
#include <filesystem>    // for filesystem, weakly_canonical
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

std::vector<char> LoadSequentialFile(std::string uri) {
  auto OpenErr = [&uri]() {
    std::string msg;
    msg = "Opening " + uri + " failed: ";
    msg += SystemErrorMsg();
    LOG(FATAL) << msg;
  };

  auto parsed = dmlc::io::URI(uri.c_str());
  CHECK((parsed.protocol == "file://" || parsed.protocol.length() == 0))
      << "Only local file is supported.";
  // Read from file.
  auto path = std::filesystem::weakly_canonical(std::filesystem::u8path(uri));
  std::ifstream ifs(path, std::ios_base::binary | std::ios_base::in);
  if (!ifs) {
    // https://stackoverflow.com/a/17338934
    OpenErr();
  }

  auto file_size = std::filesystem::file_size(path);
  std::vector<char> buffer(file_size);
  ifs.read(&buffer[0], file_size);

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

// For some reason, NVCC 12.1 marks the function deleted if we expose it in the header.
// NVCC 11.8 doesn't allow `noexcept(false) = default` altogether.
ResourceHandler::~ResourceHandler() noexcept(false) {}  // NOLINT

struct MMAPFile {
#if defined(xgboost_IS_WIN)
  HANDLE fd{INVALID_HANDLE_VALUE};
  HANDLE file_map{INVALID_HANDLE_VALUE};
#else
  std::int32_t fd{0};
#endif
  std::byte* base_ptr{nullptr};
  std::size_t base_size{0};
  std::size_t delta{0};
  std::string path;

  MMAPFile() = default;

#if defined(xgboost_IS_WIN)
  MMAPFile(HANDLE fd, HANDLE fm, std::byte* base_ptr, std::size_t base_size, std::size_t delta,
           std::string path)
      : fd{fd},
        file_map{fm},
        base_ptr{base_ptr},
        base_size{base_size},
        delta{delta},
        path{std::move(path)} {}
#else
  MMAPFile(std::int32_t fd, std::byte* base_ptr, std::size_t base_size, std::size_t delta,
           std::string path)
      : fd{fd}, base_ptr{base_ptr}, base_size{base_size}, delta{delta}, path{std::move(path)} {}
#endif
};

std::unique_ptr<MMAPFile> Open(std::string path, std::size_t offset, std::size_t length) {
  if (length == 0) {
    return std::make_unique<MMAPFile>();
  }

#if defined(xgboost_IS_WIN)
  HANDLE fd = CreateFile(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                         FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);
  CHECK_NE(fd, INVALID_HANDLE_VALUE) << "Failed to open:" << path << ". " << SystemErrorMsg();
#else
  auto fd = open(path.c_str(), O_RDONLY);
  CHECK_GE(fd, 0) << "Failed to open:" << path << ". " << SystemErrorMsg();
#endif

  std::byte* ptr{nullptr};
  // Round down for alignment.
  auto view_start = offset / GetMmapAlignment() * GetMmapAlignment();
  auto view_size = length + (offset - view_start);

#if defined(__linux__) || defined(__GLIBC__)
  int prot{PROT_READ};
  ptr = reinterpret_cast<std::byte*>(mmap(nullptr, view_size, prot, MAP_PRIVATE, fd, view_start));
  CHECK_NE(ptr, MAP_FAILED) << "Failed to map: " << path << ". " << SystemErrorMsg();
  madvise(ptr, view_size, MADV_WILLNEED);
  auto handle =
      std::make_unique<MMAPFile>(fd, ptr, view_size, offset - view_start, std::move(path));
#elif defined(xgboost_IS_WIN)
  auto file_size = GetFileSize(fd, nullptr);
  DWORD access = PAGE_READONLY;
  auto map_file = CreateFileMapping(fd, nullptr, access, 0, file_size, nullptr);
  access = FILE_MAP_READ;
  std::uint32_t loff = static_cast<std::uint32_t>(view_start);
  std::uint32_t hoff = view_start >> 32;
  CHECK(map_file) << "Failed to map: " << path << ". " << SystemErrorMsg();
  ptr = reinterpret_cast<std::byte*>(MapViewOfFile(map_file, access, hoff, loff, view_size));
  CHECK_NE(ptr, nullptr) << "Failed to map: " << path << ". " << SystemErrorMsg();
  auto handle = std::make_unique<MMAPFile>(fd, map_file, ptr, view_size, offset - view_start,
                                           std::move(path));
#else
  CHECK_LE(offset, std::numeric_limits<off_t>::max())
      << "File size has exceeded the limit on the current system.";
  int prot{PROT_READ};
  ptr = reinterpret_cast<std::byte*>(mmap(nullptr, view_size, prot, MAP_PRIVATE, fd, view_start));
  CHECK_NE(ptr, MAP_FAILED) << "Failed to map: " << path << ". " << SystemErrorMsg();
  auto handle =
      std::make_unique<MMAPFile>(fd, ptr, view_size, offset - view_start, std::move(path));
#endif  // defined(__linux__)

  return handle;
}

MmapResource::MmapResource(std::string path, std::size_t offset, std::size_t length)
    : ResourceHandler{kMmap}, handle_{Open(std::move(path), offset, length)}, n_{length} {}

MmapResource::~MmapResource() noexcept(false) {
  if (!handle_) {
    return;
  }
#if defined(xgboost_IS_WIN)
  if (handle_->base_ptr) {
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

[[nodiscard]] void* MmapResource::Data() {
  if (!handle_) {
    return nullptr;
  }
  return handle_->base_ptr + handle_->delta;
}

[[nodiscard]] std::size_t MmapResource::Size() const { return n_; }

// For some reason, NVCC 12.1 marks the function deleted if we expose it in the header.
// NVCC 11.8 doesn't allow `noexcept(false) = default` altogether.
AlignedResourceReadStream::~AlignedResourceReadStream() noexcept(false) {}  // NOLINT
PrivateMmapConstStream::~PrivateMmapConstStream() noexcept(false) {}        // NOLINT

AlignedFileWriteStream::AlignedFileWriteStream(StringView path, StringView flags)
    : pimpl_{dmlc::Stream::Create(path.c_str(), flags.c_str())} {}

[[nodiscard]] std::size_t AlignedFileWriteStream::DoWrite(const void* ptr,
                                                          std::size_t n_bytes) noexcept(true) {
  pimpl_->Write(ptr, n_bytes);
  return n_bytes;
}

AlignedMemWriteStream::AlignedMemWriteStream(std::string* p_buf)
    : pimpl_{std::make_unique<MemoryBufferStream>(p_buf)} {}
AlignedMemWriteStream::~AlignedMemWriteStream() = default;

[[nodiscard]] std::size_t AlignedMemWriteStream::DoWrite(const void* ptr,
                                                         std::size_t n_bytes) noexcept(true) {
  this->pimpl_->Write(ptr, n_bytes);
  return n_bytes;
}

[[nodiscard]] std::size_t AlignedMemWriteStream::Tell() const noexcept(true) {
  return this->pimpl_->Tell();
}
}  // namespace xgboost::common

#if defined(xgboost_IS_WIN)
#undef xgboost_IS_WIN
#endif  // defined(xgboost_IS_WIN)
