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
class CaptureStdRdBuf {
  std::stringstream ss_;
  std::streambuf* old_;

  void Release() {
    if (old_) {
      (is_stderr ? std::cerr : std::cout).rdbuf(old_);
      old_ = nullptr;
    }
  }

 public:
  explicit CaptureStdRdBuf() : old_{(is_stderr ? std::cerr : std::cout).rdbuf(ss_.rdbuf())} {}
  ~CaptureStdRdBuf() { this->Release(); }
  [[nodiscard]] std::string StopAndGetStr() {
    this->Release();
    return ss_.str();
  }
};

using CaptureStderr = CaptureStdRdBuf<true>;
using CaptureStdout = CaptureStdRdBuf<false>;

#else

// Use the internal capture functions from googletest.
template <bool is_stderr>
class CaptureStdGtest {
  bool released_{false};
  std::string captured_;

 public:
  explicit CaptureStdGtest() {
    is_stderr ? ::testing::internal::CaptureStderr() : ::testing::internal::CaptureStdout();
  }
  // Use the get string method to release the capture
  ~CaptureStdGtest() { this->StopAndGetStr(); }
  std::string StopAndGetStr() {
    if (!this->released_) {
      this->captured_ = (is_stderr ? ::testing::internal::GetCapturedStderr()
                                   : ::testing::internal::GetCapturedStdout());
      this->released_ = true;
    }
    return this->captured_;
  }
};

using CaptureStderr = CaptureStdGtest<true>;
using CaptureStdout = CaptureStdGtest<false>;
#endif
}  // namespace xgboost
