/**
 *  Copyright 2023-2024, XGBoost Contributors
 */
#pragma once

#include <cstdint>       // for int32_t
#include <memory>        // for unique_ptr
#include <string>        // for string
#include <system_error>  // for error_code
#include <utility>       // for move

namespace xgboost::collective {
namespace detail {
struct ResultImpl {
  std::string message;
  std::error_code errc{};  // optional for system error.

  std::unique_ptr<ResultImpl> prev{nullptr};

  ResultImpl() = delete;  // must initialize.
  ResultImpl(ResultImpl const& that) = delete;
  ResultImpl(ResultImpl&& that) = default;
  ResultImpl& operator=(ResultImpl const& that) = delete;
  ResultImpl& operator=(ResultImpl&& that) = default;

  explicit ResultImpl(std::string msg) : message{std::move(msg)} {}
  explicit ResultImpl(std::string msg, std::error_code errc)
      : message{std::move(msg)}, errc{std::move(errc)} {}
  explicit ResultImpl(std::string msg, std::unique_ptr<ResultImpl> prev)
      : message{std::move(msg)}, prev{std::move(prev)} {}
  explicit ResultImpl(std::string msg, std::error_code errc, std::unique_ptr<ResultImpl> prev)
      : message{std::move(msg)}, errc{std::move(errc)}, prev{std::move(prev)} {}

  [[nodiscard]] bool operator==(ResultImpl const& that) const noexcept(true) {
    if ((prev && !that.prev) || (!prev && that.prev)) {
      // one of them doesn't have prev
      return false;
    }

    auto cur_eq = message == that.message && errc == that.errc;
    if (prev && that.prev) {
      // recursive comparison
      auto prev_eq = *prev == *that.prev;
      return cur_eq && prev_eq;
    }
    return cur_eq;
  }

  [[nodiscard]] std::string Report() const;
  [[nodiscard]] std::error_code Code() const;

  void Concat(std::unique_ptr<ResultImpl> rhs);
};

#if (!defined(__GNUC__) && !defined(__clang__)) || defined(__MINGW32__)
#define __builtin_FILE() nullptr
#define __builtin_LINE() (-1)
#endif

std::string MakeMsg(std::string&& msg, char const* file, std::int32_t line);
}  // namespace detail

/**
 * @brief An error type that's easier to handle than throwing dmlc exception. We can
 *        record and propagate the system error code.
 */
struct Result {
 private:
  std::unique_ptr<detail::ResultImpl> impl_{nullptr};

 public:
  Result() noexcept(true) = default;
  explicit Result(std::string msg) : impl_{std::make_unique<detail::ResultImpl>(std::move(msg))} {}
  explicit Result(std::string msg, std::error_code errc)
      : impl_{std::make_unique<detail::ResultImpl>(std::move(msg), std::move(errc))} {}
  Result(std::string msg, Result&& prev)
      : impl_{std::make_unique<detail::ResultImpl>(std::move(msg), std::move(prev.impl_))} {}
  Result(std::string msg, std::error_code errc, Result&& prev)
      : impl_{std::make_unique<detail::ResultImpl>(std::move(msg), std::move(errc),
                                                   std::move(prev.impl_))} {}

  Result(Result const& that) = delete;
  Result& operator=(Result const& that) = delete;
  Result(Result&& that) = default;
  Result& operator=(Result&& that) = default;

  [[nodiscard]] bool OK() const noexcept(true) { return !impl_; }
  [[nodiscard]] std::string Report() const { return OK() ? "" : impl_->Report(); }
  /**
   * @brief Return the root system error. This might return success if there's no system error.
   */
  [[nodiscard]] auto Code() const { return OK() ? std::error_code{} : impl_->Code(); }
  [[nodiscard]] bool operator==(Result const& that) const noexcept(true) {
    if (OK() && that.OK()) {
      return true;
    }
    if ((OK() && !that.OK()) || (!OK() && that.OK())) {
      return false;
    }
    return *impl_ == *that.impl_;
  }

  friend Result operator+(Result&& lhs, Result&& rhs);
};

[[nodiscard]] inline Result operator+(Result&& lhs, Result&& rhs) {
  if (lhs.OK()) {
    return std::forward<Result>(rhs);
  }
  if (rhs.OK()) {
    return std::forward<Result>(lhs);
  }
  lhs.impl_->Concat(std::move(rhs.impl_));
  return std::forward<Result>(lhs);
}

/**
 * @brief Return success.
 */
[[nodiscard]] inline auto Success() noexcept(true) { return Result{}; }
/**
 * @brief Return failure.
 */
[[nodiscard]] inline auto Fail(std::string msg, char const* file = __builtin_FILE(),
                               std::int32_t line = __builtin_LINE()) {
  return Result{detail::MakeMsg(std::move(msg), file, line)};
}
/**
 * @brief Return failure with `errno`.
 */
[[nodiscard]] inline auto Fail(std::string msg, std::error_code errc,
                               char const* file = __builtin_FILE(),
                               std::int32_t line = __builtin_LINE()) {
  return Result{detail::MakeMsg(std::move(msg), file, line), std::move(errc)};
}
/**
 * @brief Return failure with a previous error.
 */
[[nodiscard]] inline auto Fail(std::string msg, Result&& prev, char const* file = __builtin_FILE(),
                               std::int32_t line = __builtin_LINE()) {
  return Result{detail::MakeMsg(std::move(msg), file, line), std::forward<Result>(prev)};
}
/**
 * @brief Return failure with a previous error and a new `errno`.
 */
[[nodiscard]] inline auto Fail(std::string msg, std::error_code errc, Result&& prev,
                               char const* file = __builtin_FILE(),
                               std::int32_t line = __builtin_LINE()) {
  return Result{detail::MakeMsg(std::move(msg), file, line), std::move(errc),
                std::forward<Result>(prev)};
}

// We don't have monad, a simple helper would do.
template <typename Fn>
[[nodiscard]] std::enable_if_t<std::is_invocable_v<Fn>, Result> operator<<(Result&& r, Fn&& fn) {
  if (!r.OK()) {
    return std::forward<Result>(r);
  }
  return fn();
}

void SafeColl(Result const& rc, char const* file = __builtin_FILE(),
              std::int32_t line = __builtin_LINE());
}  // namespace xgboost::collective
