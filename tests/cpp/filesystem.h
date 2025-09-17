/**
 * Copyright 2022-2025, XGBoost Contributors
 */
#ifndef XGBOOST_TESTS_CPP_FILESYSTEM_H
#define XGBOOST_TESTS_CPP_FILESYSTEM_H

#include <filesystem>  // for path

namespace xgboost::common {
class TemporaryDirectory {
  std::filesystem::path path_;
  std::string prefix_;

 public:
  explicit TemporaryDirectory(std::string prefix = "xgboost-");
  ~TemporaryDirectory() noexcept(false);

  [[nodiscard]] std::filesystem::path const& Path() const { return this->path_; }
  // Path can be implicitly converted to string on unix, but not on windows, due its use
  // of wchar.
  [[nodiscard]] std::string Str() const { return this->path_.string(); }
};
}  // namespace xgboost::common

#endif  // XGBOOST_TESTS_CPP_FILESYSTEM_H
