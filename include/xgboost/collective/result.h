/**
 *  Copyright 2023, XGBoost Contributors
 */
#pragma once

#include <memory>   // for unique_ptr
#include <sstream>  // for stringstream
#include <stack>    // for stack
#include <string>   // for string
#include <utility>  // for move

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

  [[nodiscard]] std::string Report() {
    std::stringstream ss;
    ss << "\n- " << this->message;
    if (this->errc != std::error_code{}) {
      ss << " system error:" << this->errc.message();
    }

    auto ptr = prev.get();
    while (ptr) {
      ss << "\n- ";
      ss << ptr->message;

      if (ptr->errc != std::error_code{}) {
        ss << " " << ptr->errc.message();
      }
      ptr = ptr->prev.get();
    }

    return ss.str();
  }
  [[nodiscard]] auto Code() const {
    // Find the root error.
    std::stack<ResultImpl const*> stack;
    auto ptr = this;
    while (ptr) {
      stack.push(ptr);
      if (ptr->prev) {
        ptr = ptr->prev.get();
      } else {
        break;
      }
    }
    while (!stack.empty()) {
      auto frame = stack.top();
      stack.pop();
      if (frame->errc != std::error_code{}) {
        return frame->errc;
      }
    }
    return std::error_code{};
  }
};
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
};

/**
 * @brief Return success.
 */
[[nodiscard]] inline auto Success() noexcept(true) { return Result{}; }
/**
 * @brief Return failure.
 */
[[nodiscard]] inline auto Fail(std::string msg) { return Result{std::move(msg)}; }
/**
 * @brief Return failure with `errno`.
 */
[[nodiscard]] inline auto Fail(std::string msg, std::error_code errc) {
  return Result{std::move(msg), std::move(errc)};
}
/**
 * @brief Return failure with a previous error.
 */
[[nodiscard]] inline auto Fail(std::string msg, Result&& prev) {
  return Result{std::move(msg), std::forward<Result>(prev)};
}
/**
 * @brief Return failure with a previous error and a new `errno`.
 */
[[nodiscard]] inline auto Fail(std::string msg, std::error_code errc, Result&& prev) {
  return Result{std::move(msg), std::move(errc), std::forward<Result>(prev)};
}
}  // namespace xgboost::collective
