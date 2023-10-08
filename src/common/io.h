/**
 * Copyright 2014-2023, XGBoost Contributors
 * \file io.h
 * \brief general stream interface for serialization, I/O
 * \author Tianqi Chen
 */
#ifndef XGBOOST_COMMON_IO_H_
#define XGBOOST_COMMON_IO_H_

#include <dmlc/io.h>
#include <rabit/rabit.h>

#include <algorithm>    // for min, fill_n, copy_n
#include <array>        // for array
#include <cstddef>      // for byte, size_t
#include <cstdlib>      // for malloc, realloc, free
#include <cstring>      // for memcpy
#include <fstream>      // for ifstream
#include <limits>       // for numeric_limits
#include <memory>       // for unique_ptr
#include <string>       // for string
#include <type_traits>  // for alignment_of_v, enable_if_t
#include <utility>      // for move
#include <vector>       // for vector

#include "common.h"
#include "xgboost/string_view.h"  // for StringView

namespace xgboost::common {
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
  [[nodiscard]] std::size_t Size() const { return buffer_.size(); }
  [[nodiscard]] std::size_t Tell() const { return pointer_; }
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

struct MMAPFile;

/**
 * @brief Handler for one-shot resource. Unlike `std::pmr::*`, the resource handler is
 *        fixed once it's constructed. Users cannot use mutable operations like resize
 *        without acquiring the specific resource first.
 */
class ResourceHandler {
 public:
  // RTTI
  enum Kind : std::uint8_t {
    kMalloc = 0,
    kMmap = 1,
  };

 private:
  Kind kind_{kMalloc};

 public:
  virtual void* Data() = 0;
  template <typename T>
  [[nodiscard]] T* DataAs() {
    return reinterpret_cast<T*>(this->Data());
  }

  [[nodiscard]] virtual std::size_t Size() const = 0;
  [[nodiscard]] auto Type() const { return kind_; }

  // Allow exceptions for cleaning up resource.
  virtual ~ResourceHandler() noexcept(false);

  explicit ResourceHandler(Kind kind) : kind_{kind} {}
  // Use shared_ptr to manage a pool like resource handler. All copy and assignment
  // operators are disabled.
  ResourceHandler(ResourceHandler const& that) = delete;
  ResourceHandler& operator=(ResourceHandler const& that) = delete;
  ResourceHandler(ResourceHandler&& that) = delete;
  ResourceHandler& operator=(ResourceHandler&& that) = delete;
  /**
   * @brief Wether two resources have the same type. (both malloc or both mmap).
   */
  [[nodiscard]] bool IsSameType(ResourceHandler const& that) const {
    return this->Type() == that.Type();
  }
};

class MallocResource : public ResourceHandler {
  void* ptr_{nullptr};
  std::size_t n_{0};

  void Clear() noexcept(true) {
    std::free(ptr_);
    ptr_ = nullptr;
    n_ = 0;
  }

 public:
  explicit MallocResource(std::size_t n_bytes) : ResourceHandler{kMalloc} { this->Resize(n_bytes); }
  ~MallocResource() noexcept(true) override { this->Clear(); }

  void* Data() override { return ptr_; }
  [[nodiscard]] std::size_t Size() const override { return n_; }
  /**
   * @brief Resize the resource to n_bytes. Unlike std::vector::resize, it prefers realloc
   *        over malloc.
   *
   * @tparam force_malloc Force the use of malloc over realloc. Used for testing.
   *
   * @param n_bytes The new size.
   */
  template <bool force_malloc = false>
  void Resize(std::size_t n_bytes, std::byte init = std::byte{0}) {
    // realloc(ptr, 0) works, but is deprecated.
    if (n_bytes == 0) {
      this->Clear();
      return;
    }

    // If realloc fails, we need to copy the data ourselves.
    bool need_copy{false};
    void* new_ptr{nullptr};
    // use realloc first, it can handle nullptr.
    if constexpr (!force_malloc) {
      new_ptr = std::realloc(ptr_, n_bytes);
    }
    // retry with malloc if realloc fails
    if (!new_ptr) {
      // ptr_ is preserved if realloc fails
      new_ptr = std::malloc(n_bytes);
      need_copy = true;
    }
    if (!new_ptr) {
      // malloc fails
      LOG(FATAL) << "bad_malloc: Failed to allocate " << n_bytes << " bytes.";
    }

    if (need_copy) {
      std::copy_n(reinterpret_cast<std::byte*>(ptr_), n_, reinterpret_cast<std::byte*>(new_ptr));
    }
    // default initialize
    std::fill_n(reinterpret_cast<std::byte*>(new_ptr) + n_, n_bytes - n_, init);
    // free the old ptr if malloc is used.
    if (need_copy) {
      this->Clear();
    }

    ptr_ = new_ptr;
    n_ = n_bytes;
  }
};

/**
 * @brief A class for wrapping mmap as a resource for RAII.
 */
class MmapResource : public ResourceHandler {
  std::unique_ptr<MMAPFile> handle_;
  std::size_t n_;

 public:
  MmapResource(std::string path, std::size_t offset, std::size_t length);
  ~MmapResource() noexcept(false) override;

  [[nodiscard]] void* Data() override;
  [[nodiscard]] std::size_t Size() const override;
};

/**
 * @param Alignment for resource read stream and aligned write stream.
 */
constexpr std::size_t IOAlignment() {
  // For most of the pod types in XGBoost, 8 byte is sufficient.
  return 8;
}

/**
 * @brief Wrap resource into a dmlc stream.
 *
 *  This class is to facilitate the use of mmap. Caller can optionally use the `Read()`
 *  method or the `Consume()` method. The former copies data into output, while the latter
 *  makes copy only if it's a primitive type.
 *
 *  Input is required to be aligned to IOAlignment().
 */
class AlignedResourceReadStream {
  std::shared_ptr<ResourceHandler> resource_;
  std::size_t curr_ptr_{0};

  // Similar to SEEK_END in libc
  static std::size_t constexpr kSeekEnd = std::numeric_limits<std::size_t>::max();

 public:
  explicit AlignedResourceReadStream(std::shared_ptr<ResourceHandler> resource)
      : resource_{std::move(resource)} {}

  [[nodiscard]] std::shared_ptr<ResourceHandler> Share() noexcept(true) { return resource_; }
  /**
   * @brief Consume n_bytes of data, no copying is performed.
   *
   * @return A pair with the beginning pointer and the number of available bytes, which
   *         may be smaller than requested.
   */
  [[nodiscard]] auto Consume(std::size_t n_bytes) noexcept(true) {
    auto res_size = resource_->Size();
    auto data = reinterpret_cast<std::byte*>(resource_->Data());
    auto ptr = data + curr_ptr_;

    // Move the cursor
    auto aligned_n_bytes = DivRoundUp(n_bytes, IOAlignment()) * IOAlignment();
    auto aligned_forward = std::min(res_size - curr_ptr_, aligned_n_bytes);
    std::size_t forward = std::min(res_size - curr_ptr_, n_bytes);

    curr_ptr_ += aligned_forward;

    return std::pair{ptr, forward};
  }

  template <typename T>
  [[nodiscard]] auto Consume(T* out) noexcept(false) -> std::enable_if_t<std::is_pod_v<T>, bool> {
    auto [ptr, size] = this->Consume(sizeof(T));
    if (size != sizeof(T)) {
      return false;
    }
    CHECK_EQ(reinterpret_cast<std::uintptr_t>(ptr) % std::alignment_of_v<T>, 0);
    *out = *reinterpret_cast<T*>(ptr);
    return true;
  }

  [[nodiscard]] virtual std::size_t Tell() noexcept(true) { return curr_ptr_; }
  /**
   * @brief Read n_bytes of data, output is copied into ptr.
   */
  [[nodiscard]] std::size_t Read(void* ptr, std::size_t n_bytes) noexcept(true) {
    auto [res_ptr, forward] = this->Consume(n_bytes);
    if (forward != 0) {
      std::memcpy(ptr, res_ptr, forward);
    }
    return forward;
  }
  /**
   * @brief Read a primitive type.
   *
   * @return Whether the read is successful.
   */
  template <typename T>
  [[nodiscard]] auto Read(T* out) noexcept(false) -> std::enable_if_t<std::is_pod_v<T>, bool> {
    return this->Consume(out);
  }
  /**
   * @brief Read a vector.
   *
   * @return Whether the read is successful.
   */
  template <typename T>
  [[nodiscard]] bool Read(std::vector<T>* out) noexcept(true) {
    std::uint64_t n{0};
    if (!this->Consume(&n)) {
      return false;
    }
    out->resize(n);

    auto n_bytes = sizeof(T) * n;
    if (this->Read(out->data(), n_bytes) != n_bytes) {
      return false;
    }
    return true;
  }

  virtual ~AlignedResourceReadStream() noexcept(false);
};

/**
 * @brief Private mmap file as a read-only stream.
 *
 *  It can calculate alignment automatically based on system page size (or allocation
 *  granularity on Windows).
 *
 *  The file is required to be aligned by IOAlignment().
 */
class PrivateMmapConstStream : public AlignedResourceReadStream {
 public:
  /**
   * @brief Construct a private mmap stream.
   *
   * @param path      File path.
   * @param offset    See the `offset` parameter of `mmap` for details.
   * @param length    See the `length` parameter of `mmap` for details.
   */
  explicit PrivateMmapConstStream(std::string path, std::size_t offset, std::size_t length)
      : AlignedResourceReadStream{std::shared_ptr<MmapResource>{  // NOLINT
            new MmapResource{std::move(path), offset, length}}} {}
  ~PrivateMmapConstStream() noexcept(false) override;
};

/**
 * @brief Base class for write stream with alignment defined by IOAlignment().
 */
class AlignedWriteStream {
 protected:
  [[nodiscard]] virtual std::size_t DoWrite(const void* ptr,
                                            std::size_t n_bytes) noexcept(true) = 0;

 public:
  virtual ~AlignedWriteStream() = default;

  [[nodiscard]] std::size_t Write(const void* ptr, std::size_t n_bytes) noexcept(false) {
    auto aligned_n_bytes = DivRoundUp(n_bytes, IOAlignment()) * IOAlignment();
    auto w_n_bytes = this->DoWrite(ptr, n_bytes);
    CHECK_EQ(w_n_bytes, n_bytes);
    auto remaining = aligned_n_bytes - n_bytes;
    if (remaining > 0) {
      std::array<std::uint8_t, IOAlignment()> padding;
      std::memset(padding.data(), '\0', padding.size());
      w_n_bytes = this->DoWrite(padding.data(), remaining);
      CHECK_EQ(w_n_bytes, remaining);
    }
    return aligned_n_bytes;
  }

  template <typename T>
  [[nodiscard]] std::enable_if_t<std::is_pod_v<T>, std::size_t> Write(T const& v) {
    return this->Write(&v, sizeof(T));
  }
};

/**
 * @brief Output stream backed by a file. Aligned to IOAlignment() bytes.
 */
class AlignedFileWriteStream : public AlignedWriteStream {
  std::unique_ptr<dmlc::Stream> pimpl_;

 protected:
  [[nodiscard]] std::size_t DoWrite(const void* ptr, std::size_t n_bytes) noexcept(true) override;

 public:
  AlignedFileWriteStream() = default;
  AlignedFileWriteStream(StringView path, StringView flags);
  ~AlignedFileWriteStream() override = default;
};

/**
 * @brief Output stream backed by memory buffer. Aligned to IOAlignment() bytes.
 */
class AlignedMemWriteStream : public AlignedFileWriteStream {
  std::unique_ptr<MemoryBufferStream> pimpl_;

 protected:
  [[nodiscard]] std::size_t DoWrite(const void* ptr, std::size_t n_bytes) noexcept(true) override;

 public:
  explicit AlignedMemWriteStream(std::string* p_buf);
  ~AlignedMemWriteStream() override;

  [[nodiscard]] std::size_t Tell() const noexcept(true);
};
}  // namespace xgboost::common
#endif  // XGBOOST_COMMON_IO_H_
