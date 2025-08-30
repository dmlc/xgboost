/**
 * Copyright 2022-2025, XGBoost Contributors
 */
#ifndef XGBOOST_TESTS_CPP_FILESYSTEM_H
#define XGBOOST_TESTS_CPP_FILESYSTEM_H

#include <xgboost/windefs.h>

#include <filesystem>  // for path

#include "dmlc/filesystem.h"

namespace xgboost::common {
class TemporaryDirectory {
  std::filesystem::path path_;

 public:
  TemporaryDirectory();
  ~TemporaryDirectory() noexcept(false);

  [[nodiscard]] std::filesystem::path const& Path() const { return this->path_; }
};
}  // namespace xgboost::common

#endif  // XGBOOST_TESTS_CPP_FILESYSTEM_H
