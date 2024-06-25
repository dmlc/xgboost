/**
 *  Copyright 2021-2024, XGBoost Contributors
 */
#include "sparse_page_source.h"

#include <filesystem>  // for exists

namespace xgboost::data {
void TryDeleteCacheFile(const std::string& file) {
  // Don't throw, this is called in a destructor.
  auto exists = std::filesystem::exists(file);
  if (!exists) {
    LOG(WARNING) << "External memory cache file " << file << " is missing.";
  }
  if (std::remove(file.c_str()) != 0) {
    LOG(WARNING) << "Couldn't remove external memory cache file " << file
                 << "; you may want to remove it manually";
  }
}
}  // namespace xgboost::data
