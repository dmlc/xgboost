/**
 * Copyright 2019-2025, by XGBoost Contributors
 */
#include "error_msg.h"
#if defined(__unix__) || defined(__APPLE__)

#include <fcntl.h>     // for open, O_RDONLY, posix_fadvise
#include <sys/mman.h>  // for mmap, munmap, madvise
#include <unistd.h>    // for close, getpagesize

#else

#include <xgboost/windefs.h>

#if defined(xgboost_IS_WIN)

#include <windows.h>  // for CreateFileMapping2, CreateFileEx...

#endif  // defined(xgboost_IS_WIN)

#endif  // defined(__unix__) || defined(__APPLE__)

#include <algorithm>     // for copy, transform
#include <cctype>        // for tolower
#include <cstddef>       // for size_t
#include <cstdint>       // for int32_t, uint32_t
#include <cstdio>        // for fread, fseek
#include <cstring>       // for memcpy
#include <filesystem>    // for filesystem, weakly_canonical
#include <fstream>       // for ifstream
#include <iterator>      // for distance
#include <memory>        // for unique_ptr, make_unique
#include <string>        // for string
#include <utility>       // for move
#include <vector>        // for vector

#include "io.h"
#include "xgboost/logging.h"            // for CHECK_LE
#include "xgboost/string_view.h"        // for StringView

#if !defined(__linux__) && !defined(__GLIBC__) && !defined(xgboost_IS_WIN)
#include <limits>  // for numeric_limits
#endif

#if defined(__linux__)
#include <sys/sysinfo.h>
#endif

namespace xgboost::common {
size_t PeekableInStream::Read(void* dptr, size_t size) {
  size_t nbuffer = buffer_.length() - buffer_ptr_;
  if (nbuffer == 0) return strm_->Read(dptr, size);
  if (nbuffer < size) {
    std::memcpy(dptr, dmlc::BeginPtr(buffer_) + buffer_ptr_, nbuffer);
    buffer_ptr_ += nbuffer;
    return nbuffer + strm_->Read(reinterpret_cast<char*>(dptr) + nbuffer, size - nbuffer);
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
  if (size >= buffer_.size() - pointer_) {
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
}  // anonymous namespace

std::vector<char> LoadSequentialFile(std::string uri) {
  auto OpenErr = [&uri]() {
    std::string msg;
    msg = "Opening " + uri + " failed: ";
    msg += error::SystemError().message();
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

struct MmapFileImpl {
#if defined(xgboost_IS_WIN)
  HANDLE fd{INVALID_HANDLE_VALUE};
  HANDLE file_map{INVALID_HANDLE_VALUE};
#else
  std::int32_t fd{0};
#endif  // defined(xgboost_IS_WIN)
  std::byte* base_ptr{nullptr};
  std::size_t base_size{0};
  std::size_t delta{0};
  std::string path;

  MmapFileImpl() = default;

#if defined(xgboost_IS_WIN)
  MmapFileImpl(HANDLE fd, HANDLE fm, std::byte* base_ptr, std::size_t base_size, std::size_t delta,
               std::string path)
      : fd{fd},
        file_map{fm},
        base_ptr{base_ptr},
        base_size{base_size},
        delta{delta},
        path{std::move(path)} {}
#else
  MmapFileImpl(std::int32_t fd, std::byte* base_ptr, std::size_t base_size, std::size_t delta,
               std::string path)
      : fd{fd}, base_ptr{base_ptr}, base_size{base_size}, delta{delta}, path{std::move(path)} {}
#endif  // defined(xgboost_IS_WIN)

  void const* Data() const { return this->base_ptr + this->delta; }
  void* Data() { return this->base_ptr + this->delta; }
};

void const* MMAPFile::Data() const {
  if (!this->p_impl) {
    return nullptr;
  }
  return this->p_impl->Data();
}

void* MMAPFile::Data() {
  if (!this->p_impl) {
    return nullptr;
  }
  return this->p_impl->Data();
}

[[nodiscard]] Span<std::byte> MMAPFile::BasePtr() const {
  return Span{this->p_impl->base_ptr, this->p_impl->base_size};
}

// For some reason, NVCC 12.1 marks the function deleted if we expose it in the header.
// NVCC 11.8 doesn't allow `noexcept(false) = default` altogether.
ResourceHandler::~ResourceHandler() noexcept(false) {}  // NOLINT

MMAPFile* detail::OpenMmap(std::string path, std::size_t offset, std::size_t length) {
  if (length == 0) {
    return new MMAPFile{};
  }

#if defined(xgboost_IS_WIN)
  HANDLE fd = CreateFile(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                         FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);
  CHECK_NE(fd, INVALID_HANDLE_VALUE)
      << "Failed to open:" << path << ". " << error::SystemError().message();
#else
  auto fd = open(path.c_str(), O_RDONLY);
  CHECK_GE(fd, 0) << "Failed to open:" << path << ". " << error::SystemError().message();
#endif

  std::byte* ptr{nullptr};
  // Round down for alignment.
  auto view_start = offset / GetMmapAlignment() * GetMmapAlignment();
  auto view_size = length + (offset - view_start);

#if defined(__linux__) || defined(__GLIBC__)
  int prot{PROT_READ};
  ptr = reinterpret_cast<std::byte*>(mmap(nullptr, view_size, prot, MAP_PRIVATE, fd, view_start));
  CHECK_NE(ptr, MAP_FAILED) << "Failed to map: " << path << ". " << error::SystemError().message();
  auto handle = new MMAPFile{
      std::make_unique<MmapFileImpl>(fd, ptr, view_size, offset - view_start, std::move(path))};
#elif defined(xgboost_IS_WIN)
  LARGE_INTEGER file_size;
  CHECK_NE(GetFileSizeEx(fd, &file_size), 0) << error::SystemError().message();
  auto map_file = CreateFileMappingA(fd, nullptr, PAGE_READONLY, file_size.HighPart,
                                     file_size.LowPart, nullptr);
  CHECK(map_file) << "Failed to map: " << path << ". " << error::SystemError().message();

  auto li_vs = reinterpret_cast<LARGE_INTEGER*>(&view_start);
  ptr = reinterpret_cast<std::byte*>(
      MapViewOfFile(map_file, FILE_MAP_READ, li_vs->HighPart, li_vs->LowPart, view_size));
  CHECK_NE(ptr, nullptr) << "Failed to map: " << path << ". " << error::SystemError().message();
  auto handle = new MMAPFile{std::make_unique<MmapFileImpl>(fd, map_file, ptr, view_size,
                                                            offset - view_start, std::move(path))};
#else
  CHECK_LE(offset, std::numeric_limits<off_t>::max())
      << "File size has exceeded the limit on the current system.";
  int prot{PROT_READ};
  ptr = reinterpret_cast<std::byte*>(mmap(nullptr, view_size, prot, MAP_PRIVATE, fd, view_start));
  CHECK_NE(ptr, MAP_FAILED) << "Failed to map: " << path << ". " << error::SystemError().message();
  auto handle = new MMAPFile{
      std::make_unique<MmapFileImpl>(fd, ptr, view_size, offset - view_start, std::move(path))};
#endif  // defined(__linux__) || defined(__GLIBC__)

  return handle;
}

void detail::CloseMmap(MMAPFile* handle) {
  if (!handle) {
    return;
  }
#if defined(xgboost_IS_WIN)
  if (handle->p_impl->base_ptr) {
    CHECK(UnmapViewOfFile(handle->p_impl->base_ptr))
        << "Failed to call munmap: " << error::SystemError().message();
  }
  if (handle->p_impl->fd != INVALID_HANDLE_VALUE) {
    CHECK(CloseHandle(handle->p_impl->fd))
        << "Failed to close handle: " << error::SystemError().message();
  }
  if (handle->p_impl->file_map != INVALID_HANDLE_VALUE) {
    CHECK(CloseHandle(handle->p_impl->file_map))
        << "Failed to close mapping object: " << error::SystemError().message();
  }
#else
  if (handle->p_impl->base_ptr) {
    CHECK_NE(munmap(handle->p_impl->base_ptr, handle->p_impl->base_size), -1)
        << "Failed to call munmap: `" << handle->p_impl->path << "`. "
        << error::SystemError().message();
  }
  if (handle->p_impl->fd != 0) {
    CHECK_NE(close(handle->p_impl->fd), -1)
        << "Failed to close: `" << handle->p_impl->path << "`. " << error::SystemError().message();
  }
#endif
  delete handle;
}

MmapResource::MmapResource(StringView path, std::size_t offset, std::size_t length)
    : ResourceHandler{kMmap},
      handle_{detail::OpenMmap(std::string{path}, offset, length), detail::CloseMmap},
      n_{length} {
#if defined(__unix__) || defined(__APPLE__)
  madvise(handle_->p_impl->base_ptr, handle_->p_impl->base_size, MADV_WILLNEED);
#endif  // defined(__unix__) || defined(__APPLE__)
}

MmapResource::~MmapResource() noexcept(false) = default;

[[nodiscard]] void* MmapResource::Data() {
  if (!handle_) {
    return nullptr;
  }
  return this->handle_->Data();
}

[[nodiscard]] std::size_t MmapResource::Size() const { return n_; }

// For some reason, NVCC 12.1 marks the function deleted if we expose it in the header.
// NVCC 11.8 doesn't allow `noexcept(false) = default` altogether.
AlignedResourceReadStream::~AlignedResourceReadStream() noexcept(false) {}  // NOLINT
PrivateMmapConstStream::~PrivateMmapConstStream() noexcept(false) {}        // NOLINT

std::shared_ptr<MallocResource> MemBufFileReadStream::ReadFileIntoBuffer(StringView path,
                                                                         std::size_t offset,
                                                                         std::size_t length) {
  CHECK(std::filesystem::exists(path.c_str())) << "`" << path << "` doesn't exist";
  auto res = std::make_shared<MallocResource>(length);
  auto ptr = res->DataAs<char>();
  std::unique_ptr<FILE, std::function<int(FILE*)>> fp{fopen(path.c_str(), "rb"), fclose};

  auto err = [&] {
    auto e = error::SystemError().message();
    LOG(FATAL) << "Failed to read file `" << path << "`. System error message: " << e;
  };
#if defined(__linux__)
  auto fd = fileno(fp.get());
  if (fd == -1) {
    err();
  }
  if (posix_fadvise(fd, offset, length, POSIX_FADV_SEQUENTIAL) != 0) {
    LOG(FATAL) << error::SystemError().message();
  }
#endif  // defined(__linux__)

  if (fseek(fp.get(), offset, SEEK_SET) != 0) {
    err();
  }
  if (fread(ptr, length, 1, fp.get()) != 1) {
    err();
  }
  return res;
}

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

[[nodiscard]] std::string CmdOutput(StringView cmd) {
#if defined(xgboost_IS_WIN)
  std::unique_ptr<FILE, std::function<int(FILE*)>> pipe(_popen(cmd.c_str(), "r"), _pclose);
#else
  // popen is a convenient method, but it always returns a success even if the command
  // fails.
  std::unique_ptr<FILE, std::function<int(FILE*)>> pipe(popen(cmd.c_str(), "r"), pclose);
#endif
  CHECK(pipe);
  std::array<char, 128> buffer;
  std::string result;
  while (std::fgets(buffer.data(), static_cast<std::int32_t>(buffer.size()), pipe.get())) {
    result += buffer.data();
  }
  return result;
}

[[nodiscard]] std::size_t TotalMemory() {
#if defined(__linux__)
  struct sysinfo info;
  CHECK_EQ(sysinfo(&info), 0) << error::SystemError().message();
  return info.totalram * info.mem_unit;
#elif defined(xgboost_IS_WIN)
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  CHECK(GlobalMemoryStatusEx(&status)) << error::SystemError().message();
  return static_cast<std::size_t>(status.ullTotalPhys);
#else
  LOG(FATAL) << "Not implemented";
#endif  // defined(__linux__)
}
}  // namespace xgboost::common
