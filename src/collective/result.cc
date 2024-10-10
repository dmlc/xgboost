/**
 *  Copyright 2024, XGBoost Contributors
 */
#include "xgboost/collective/result.h"

#include <filesystem>  // for path
#include <sstream>     // for stringstream
#include <stack>       // for stack

#include "xgboost/logging.h"

namespace xgboost::collective {
namespace detail {
[[nodiscard]] std::string ResultImpl::Report() const {
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

[[nodiscard]] std::error_code ResultImpl::Code() const {
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

void ResultImpl::Concat(std::unique_ptr<ResultImpl> rhs) {
  auto ptr = this;
  while (ptr->prev) {
    ptr = ptr->prev.get();
  }
  ptr->prev = std::move(rhs);
}

std::string MakeMsg(std::string&& msg, char const* file, std::int32_t line) {
  dmlc::DateLogger logger;
  if (file && line != -1) {
    auto name = std::filesystem::path{ file }.filename();
    return "[" + name.string() + ":" + std::to_string(line) + "|" + logger.HumanDate() +
           "]: " + std::forward<std::string>(msg);
  }
  return std::string{"["} + logger.HumanDate() + "]" + std::forward<std::string>(msg);  // NOLINT
}
}  // namespace detail

void SafeColl(Result const& rc) {
  if (!rc.OK()) {
    LOG(FATAL) << rc.Report();
  }
}
}  // namespace xgboost::collective
