/**
 * Copyright 2025, XGBoost Contributors
 *
 * @brief Helpers for capturing standard outputs.
 */
#pragma once
#include <xgboost/windefs.h>

#if defined(xgboost_IS_WIN)
#include <sstream>    // for stringstream
#include <streambuf>  // for streambuf
#else
#include <gtest/gtest.h>
#endif  // defined(xgboost_IS_WIN)

#include <string>  // for string

namespace xgboost {
#if defined(xgboost_IS_WIN)
// Custom implementation for Windows. This assumes all writes are using the C++ streams
// and doesn't work with system file descriptors.

template <bool is_stderr>
class CaptureStderrRdBufImpl {
  std::stringstream ss_;
  std::streambuf* old_;

  void Release() {
    if (old_) {
      (is_stderr ? std::cerr : std::cout).rdbuf(old_);
      old_ = nullptr;
    }
  }

 public:
  explicit CaptureStderrRdBufImpl()
      : old_{(is_stderr ? std::cerr : std::cout).rdbuf(ss_.rdbuf())} {}
  ~CaptureStderrRdBufImpl() { this->Release(); }
  [[nodiscard]] std::string GetString() {
    this->Release();
    return ss_.str();
  }
};

using CaptureStderr = CaptureStderrRdBufImpl<true>;
using CaptureStdout = CaptureStderrRdBufImpl<false>;

#else

// Use the internal capture from googletest.
class CaptureStderrGtest {
 public:
  explicit CaptureStderrGtest() { ::testing::internal::CaptureStderr(); }
  // Use the get string method to release the capture
  ~CaptureStderrGtest() { this->GetString(); }
  std::string GetString() const { return ::testing::internal::GetCapturedStderr(); }
};

class CaptureStdoutGtest {
 public:
  explicit CaptureStdoutGtest() { ::testing::internal::CaptureStdout(); }
  // Use the get string method to release the capture
  ~CaptureStdoutGtest() { this->GetString(); }
  std::string GetString() const { return ::testing::internal::GetCapturedStdout(); }
};

using CaptureStderr = CaptureStderrGtest;
using CaptureStdout = CaptureStdoutGtest;
#endif
}  // namespace xgboost
