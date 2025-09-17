/**
 * Copyright 2021-2024, XGBoost contributors
 */
#ifndef XGBOOST_DATA_FILE_ITERATOR_H_
#define XGBOOST_DATA_FILE_ITERATOR_H_

#include <cstdint>    // for uint32_t
#include <memory>     // for unique_ptr
#include <string>     // for string
#include <utility>    // for move

#include "dmlc/data.h"        // for RowBlock, Parser
#include "xgboost/c_api.h"    // for XGDMatrixFree, XGProxyDMatrixCreate

namespace xgboost::data {
[[nodiscard]] std::string ValidateFileFormat(std::string const& uri);

/**
 * An iterator for implementing external memory support with file inputs.  Users of
 * external memory are encouraged to define their own file parsers/loaders so this one is
 * just here for compatibility with old versions of XGBoost and CLI interface.
 */
class FileIterator {
  // uri of input file, encodes parameters about whether it's 1-based index etc.  dmlc
  // parser will decode these information.
  std::string uri_;
  // Equals to rank_id in distributed training, used to split file into parts for each
  // worker.
  uint32_t part_idx_;
  // Equals to total number of workers.
  uint32_t n_parts_;

  DMatrixHandle proxy_;

  std::unique_ptr<dmlc::Parser<uint32_t>> parser_;
  // Temporary reference to stage the data.
  dmlc::RowBlock<uint32_t, float> row_block_;
  // Storage for the array interface strings.
  std::string indptr_;
  std::string values_;
  std::string indices_;

 public:
  FileIterator(std::string uri, unsigned part_index, unsigned num_parts)
      : uri_{ValidateFileFormat(std::move(uri))}, part_idx_{part_index}, n_parts_{num_parts} {
    XGProxyDMatrixCreate(&proxy_);
  }
  ~FileIterator() {
    XGDMatrixFree(proxy_);
  }

  int Next();

  auto Proxy() -> decltype(proxy_) { return proxy_; }

  void Reset() {
    parser_.reset(dmlc::Parser<uint32_t>::Create(uri_.c_str(), part_idx_, n_parts_, "auto"));
  }
};

namespace fileiter {
inline void Reset(DataIterHandle self) {
  static_cast<FileIterator*>(self)->Reset();
}

inline int Next(DataIterHandle self) {
  return static_cast<FileIterator*>(self)->Next();
}
}  // namespace fileiter
}  // namespace xgboost::data
#endif  // XGBOOST_DATA_FILE_ITERATOR_H_
