/**
 *  Copyright 2021-2024, XGBoost Contributors
 */
#include "sparse_page_source.h"

#include <cstdio>      // for remove
#include <filesystem>  // for exists
#include <numeric>     // for partial_sum
#include <string>      // for string

#include "../collective/communicator-inl.h"  // for IsDistributed, GetRank

namespace xgboost::data {
void Cache::Commit() {
  if (!this->written) {
    std::partial_sum(this->offset.begin(), this->offset.end(), this->offset.begin());
    this->written = true;
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

std::string MakeCachePrefix(std::string cache_prefix) {
  cache_prefix = cache_prefix.empty() ? "DMatrix" : cache_prefix;
  if (collective::IsDistributed()) {
    cache_prefix += ("-r" + std::to_string(collective::GetRank()));
  }
  return cache_prefix;
}
}  // namespace xgboost::data
