/*!
 * Copyright 2021 XGBoost contributors
 */
#ifndef XGBOOST_DATA_FILE_ITERATOR_H_
#define XGBOOST_DATA_FILE_ITERATOR_H_

#include <string>
#include <memory>
#include <vector>
#include <utility>

#include "dmlc/data.h"
#include "xgboost/c_api.h"
#include "xgboost/json.h"
#include "xgboost/linalg.h"
#include "array_interface.h"

namespace xgboost {
namespace data {
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
  // Format of the input file, like "libsvm".
  std::string type_;

  DMatrixHandle proxy_;

  std::unique_ptr<dmlc::Parser<uint32_t>> parser_;
  // Temporary reference to stage the data.
  dmlc::RowBlock<uint32_t, float> row_block_;
  // Storage for the array interface strings.
  std::string indptr_;
  std::string values_;
  std::string indices_;

 public:
  FileIterator(std::string uri, unsigned part_index, unsigned num_parts,
               std::string type)
      : uri_{std::move(uri)}, part_idx_{part_index}, n_parts_{num_parts},
        type_{std::move(type)} {
    XGProxyDMatrixCreate(&proxy_);
  }
  ~FileIterator() {
    XGDMatrixFree(proxy_);
  }

  int Next() {
    CHECK(parser_);
    if (parser_->Next()) {
      row_block_ = parser_->Value();
      using linalg::MakeVec;

      indptr_ = ArrayInterfaceStr(MakeVec(row_block_.offset, row_block_.size + 1));
      values_ = ArrayInterfaceStr(MakeVec(row_block_.value, row_block_.offset[row_block_.size]));
      indices_ = ArrayInterfaceStr(MakeVec(row_block_.index, row_block_.offset[row_block_.size]));

      size_t n_columns = *std::max_element(row_block_.index,
                                           row_block_.index + row_block_.offset[row_block_.size]);
      // dmlc parser converts 1-based indexing back to 0-based indexing so we can ignore
      // this condition and just add 1 to n_columns
      n_columns += 1;

      XGProxyDMatrixSetDataCSR(proxy_, indptr_.c_str(), indices_.c_str(),
                               values_.c_str(), n_columns);

      if (row_block_.label) {
        XGDMatrixSetDenseInfo(proxy_, "label", row_block_.label, row_block_.size, 1);
      }
      if (row_block_.qid) {
        XGDMatrixSetDenseInfo(proxy_, "qid", row_block_.qid, row_block_.size, 1);
      }
      if (row_block_.weight) {
        XGDMatrixSetDenseInfo(proxy_, "weight", row_block_.weight, row_block_.size, 1);
      }
      // Continue iteration
      return true;
    } else {
      // Stop iteration
      return false;
    }
  }

  auto Proxy() -> decltype(proxy_) { return proxy_; }

  void Reset() {
    CHECK(!type_.empty());
    parser_.reset(dmlc::Parser<uint32_t>::Create(uri_.c_str(), part_idx_,
                                                 n_parts_, type_.c_str()));
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
}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_FILE_ITERATOR_H_
