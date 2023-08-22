/**
 * Copyright 2023, XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_REF_RESOURCE_VIEW_H_
#define XGBOOST_COMMON_REF_RESOURCE_VIEW_H_

#include <algorithm>    // for fill_n
#include <cstdint>      // for uint64_t
#include <cstring>      // for memcpy
#include <memory>       // for shared_ptr, make_shared
#include <type_traits>  // for is_reference_v, remove_reference_t, is_same_v
#include <utility>      // for swap, move

#include "io.h"  // for ResourceHandler, AlignedResourceReadStream, MallocResource
#include "xgboost/logging.h"
#include "xgboost/span.h"  // for Span

namespace xgboost::common {
/**
 * @brief A vector-like type that holds a reference counted resource.
 *
 *    The vector size is immutable after construction. This way we can swap the underlying
 *    resource when needed.
 */
template <typename T>
class RefResourceView {
  static_assert(!std::is_reference_v<T>);

 public:
  using value_type = T;             // NOLINT
  using size_type = std::uint64_t;  // NOLINT

 private:
  value_type* ptr_{nullptr};
  size_type size_{0};
  std::shared_ptr<common::ResourceHandler> mem_{nullptr};

 protected:
  void Init(value_type* ptr, size_type size, std::shared_ptr<common::ResourceHandler> mem) {
    ptr_ = ptr;
    size_ = size;
    mem_ = std::move(mem);
  }

 public:
  RefResourceView(value_type* ptr, size_type n, std::shared_ptr<common::ResourceHandler> mem)
      : ptr_{ptr}, size_{n}, mem_{std::move(mem)} {
    CHECK_GE(mem_->Size(), n);
  }
  /**
   * @brief Construct a view on ptr with length n. The ptr is held by the mem resource.
   *
   * @param ptr  The pointer to view.
   * @param n    The length of the view.
   * @param mem  The owner of the pointer.
   * @param init Initialize the view with this value.
   */
  RefResourceView(value_type* ptr, size_type n, std::shared_ptr<common::ResourceHandler> mem,
                  T const& init)
      : RefResourceView{ptr, n, mem} {
    if (n != 0) {
      std::fill_n(ptr_, n, init);
    }
  }

  ~RefResourceView() = default;

  RefResourceView() = default;
  RefResourceView(RefResourceView const& that) = delete;
  RefResourceView& operator=(RefResourceView const& that) = delete;
  /**
   * @brief We allow move assignment for lazy initialization.
   */
  RefResourceView(RefResourceView&& that) = default;
  RefResourceView& operator=(RefResourceView&& that) = default;

  [[nodiscard]] size_type size() const { return size_; }  // NOLINT
  [[nodiscard]] size_type size_bytes() const {            // NOLINT
    return Span{data(), size()}.size_bytes();
  }
  [[nodiscard]] value_type* data() { return ptr_; };              // NOLINT
  [[nodiscard]] value_type const* data() const { return ptr_; };  // NOLINT
  [[nodiscard]] bool empty() const { return size() == 0; }        // NOLINT

  [[nodiscard]] auto cbegin() const { return data(); }         // NOLINT
  [[nodiscard]] auto begin() { return data(); }                // NOLINT
  [[nodiscard]] auto begin() const { return cbegin(); }        // NOLINT
  [[nodiscard]] auto cend() const { return data() + size(); }  // NOLINT
  [[nodiscard]] auto end() { return data() + size(); }         // NOLINT
  [[nodiscard]] auto end() const { return cend(); }            // NOLINT

  [[nodiscard]] auto const& front() const { return data()[0]; }          // NOLINT
  [[nodiscard]] auto& front() { return data()[0]; }                      // NOLINT
  [[nodiscard]] auto const& back() const { return data()[size() - 1]; }  // NOLINT
  [[nodiscard]] auto& back() { return data()[size() - 1]; }              // NOLINT

  [[nodiscard]] value_type& operator[](size_type i) { return ptr_[i]; }
  [[nodiscard]] value_type const& operator[](size_type i) const { return ptr_[i]; }

  /**
   * @brief Get the underlying resource.
   */
  auto Resource() const { return mem_; }
};

/**
 * @brief Read a vector from stream. Accepts both `std::vector` and `RefResourceView`.
 *
 *  If the output vector is a referenced counted view, no copying occur.
 */
template <typename Vec>
[[nodiscard]] bool ReadVec(common::AlignedResourceReadStream* fi, Vec* vec) {
  std::uint64_t n{0};
  if (!fi->Read(&n)) {
    return false;
  }
  if (n == 0) {
    return true;
  }

  using T = typename Vec::value_type;
  auto expected_bytes = sizeof(T) * n;

  auto [ptr, n_bytes] = fi->Consume(expected_bytes);
  if (n_bytes != expected_bytes) {
    return false;
  }

  if constexpr (std::is_same_v<Vec, RefResourceView<T>>) {
    *vec = RefResourceView<T>{reinterpret_cast<T*>(ptr), n, fi->Share()};
  } else {
    vec->resize(n);
    std::memcpy(vec->data(), ptr, n_bytes);
  }
  return true;
}

/**
 * @brief Write a vector to stream. Accepts both `std::vector` and `RefResourceView`.
 */
template <typename Vec>
[[nodiscard]] std::size_t WriteVec(AlignedFileWriteStream* fo, Vec const& vec) {
  std::size_t bytes{0};
  auto n = static_cast<std::uint64_t>(vec.size());
  bytes += fo->Write(n);
  if (n == 0) {
    return sizeof(n);
  }

  using T = typename std::remove_reference_t<decltype(vec)>::value_type;
  bytes += fo->Write(vec.data(), vec.size() * sizeof(T));

  return bytes;
}

/**
 * @brief Make a fixed size `RefResourceView` with malloc resource.
 */
template <typename T>
[[nodiscard]] RefResourceView<T> MakeFixedVecWithMalloc(std::size_t n_elements, T const& init) {
  auto resource = std::make_shared<common::MallocResource>(n_elements * sizeof(T));
  return RefResourceView{resource->DataAs<T>(), n_elements, resource, init};
}

template <typename T>
class ReallocVector : public RefResourceView<T> {
  static_assert(!std::is_reference_v<T>);
  static_assert(!std::is_const_v<T>);
  static_assert(std::is_trivially_copyable_v<T>);

  using Upper = RefResourceView<T>;
  using size_type = typename Upper::size_type;    // NOLINT
  using value_type = typename Upper::value_type;  // NOLINT

 public:
  ReallocVector() : RefResourceView<T>{MakeFixedVecWithMalloc(0, T{})} {}

  ReallocVector(size_type n, value_type const& init)
      : RefResourceView<T>{MakeFixedVecWithMalloc(n, init)} {}
  ReallocVector(ReallocVector const& that) = delete;
  ReallocVector(ReallocVector&& that) = delete;
  ReallocVector& operator=(ReallocVector const& that) = delete;
  ReallocVector& operator=(ReallocVector&& that) = delete;

  void Resize(typename Upper::size_type new_size) {
    auto resource = std::dynamic_pointer_cast<common::MallocResource>(this->Resource());
    CHECK(resource);
    resource->Resize(new_size * sizeof(T));
    this->Init(resource->template DataAs<T>(), new_size, resource);
  }
};
}  // namespace xgboost::common
#endif  // XGBOOST_COMMON_REF_RESOURCE_VIEW_H_
