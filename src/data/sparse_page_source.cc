/**
 *  Copyright 2021-2024, XGBoost Contributors
 */
#include "sparse_page_source.h"

#include <cstdio>      // for remove
#include <filesystem>  // for exists
#include <numeric>     // for partial_sum
#include <string>      // for string

namespace xgboost::data {
void Cache::Commit() {
  if (!written) {
    std::partial_sum(offset.begin(), offset.end(), offset.begin());
    written = true;
  }
}

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

#if !defined(XGBOOST_USE_CUDA)
void InitNewThread::operator()() const { *GlobalConfigThreadLocalStore::Get() = config; }
#endif
}  // namespace xgboost::data
