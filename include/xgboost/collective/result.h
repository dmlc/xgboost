/**
 *  Copyright 2023, XGBoost Contributors
 */
#pragma once

#include <memory>   // for unique_ptr
#include <sstream>  // for stringstream
#include <string>   // for string
#include <utility>  // for move

namespace xgboost::collective {
namespace detail {
struct ResultImpl {
  std::string message;
  bool ok{true};
  std::error_code errc{};  // optional for system error.

  std::unique_ptr<ResultImpl> prev{nullptr};

  ResultImpl() = default;
  ResultImpl(ResultImpl const& that) = delete;
  ResultImpl(ResultImpl&& that) = default;
  ResultImpl& operator=(ResultImpl const& that) = delete;
  ResultImpl& operator=(ResultImpl&& that) = default;

  explicit ResultImpl(std::string msg) : message{std::move(msg)}, ok{false} {}
  explicit ResultImpl(std::string msg, std::error_code errc)
      : message{std::move(msg)}, ok{false}, errc{errc} {}
  explicit ResultImpl(std::string msg, std::unique_ptr<ResultImpl> prev)
      : message{std::move(msg)}, ok{false}, prev{std::move(prev)} {}
  explicit ResultImpl(std::string msg, std::error_code errc, std::unique_ptr<ResultImpl> prev)
      : message{std::move(msg)}, ok{false}, errc{std::move(errc)}, prev{std::move(prev)} {}

  [[nodiscard]] bool operator==(ResultImpl const& that) const {
    if (ok && that.ok) {
      // both are success
      return true;
    }
    if ((prev && !that.prev) || (!prev && that.prev)) {
      // one of them doesn't have prev
      return false;
    }

    auto cur_eq = ok == that.ok && message == that.message && errc == that.errc;
    if (prev && that.prev) {
      // recursive comparison
      auto prev_eq = *prev == *that.prev;
      return cur_eq && prev_eq;
    }
    return cur_eq;
  }

  [[nodiscard]] bool OK() const { return ok; }

  [[nodiscard]] std::string Report() {
    if (OK()) {
      return "";
    }

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
};
}  // namespace detail

// An error type that's easier to handle than throwing dmlc exception.
struct Result {
 private:
  std::unique_ptr<detail::ResultImpl> impl_;

 public:
  Result() : impl_{std::make_unique<detail::ResultImpl>()} {}
  explicit Result(std::string msg) : impl_{std::make_unique<detail::ResultImpl>(std::move(msg))} {}
  explicit Result(std::string msg, std::error_code errc)
      : impl_{std::make_unique<detail::ResultImpl>(std::move(msg), std::move(errc))} {}
  Result(std::string msg, Result&& prev)
      : impl_{std::make_unique<detail::ResultImpl>(msg, std::move(prev.impl_))} {}
  Result(std::string msg, std::error_code errc, Result&& prev)
      : impl_{std::make_unique<detail::ResultImpl>(msg, errc, std::move(prev.impl_))} {}

  Result(Result const& that) = delete;
  Result& operator=(Result const& that) = delete;
  Result(Result&& that) = default;
  Result& operator=(Result&& that) = delete;

  [[nodiscard]] bool OK() const { return impl_->OK(); }
  [[nodiscard]] std::string Report() const { return this->impl_->Report(); }
  [[nodiscard]] auto Code() const { return impl_->errc; }
  [[nodiscard]] bool operator==(Result const& that) const { return *impl_ == *that.impl_; }
};

[[nodiscard]] inline auto Success() { return Result{}; }

[[nodiscard]] inline auto Fail(std::string msg) { return Result{std::move(msg)}; }

/**
 * @brief Return failure with `errno`.
 */
[[nodiscard]] inline auto Fail(std::string msg, std::error_code errc) {
  return Result{std::move(msg), std::move(errc)};
}

[[nodiscard]] inline auto Fail(std::string msg, Result&& prev) {
  return Result{msg, std::forward<Result>(prev)};
}

[[nodiscard]] inline auto Fail(std::string msg, std::error_code errc, Result&& prev) {
  return Result{msg, std::move(errc), std::forward<Result>(prev)};
}
}  // namespace xgboost::collective
