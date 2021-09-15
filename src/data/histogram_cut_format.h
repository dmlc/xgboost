/*!
 * Copyright 2021 XGBoost contributors
 */
#ifndef XGBOOST_DATA_HISTOGRAM_CUT_FORMAT_H_
#define XGBOOST_DATA_HISTOGRAM_CUT_FORMAT_H_

#include "../common/hist_util.h"

namespace xgboost {
namespace data {
inline bool ReadHistogramCuts(common::HistogramCuts *cuts, dmlc::SeekStream *fi) {
  if (!fi->Read(&cuts->cut_values_.HostVector())) {
    return false;
  }
  if (!fi->Read(&cuts->cut_ptrs_.HostVector())) {
    return false;
  }
  if (!fi->Read(&cuts->min_vals_.HostVector())) {
    return false;
  }
  return true;
}

inline size_t WriteHistogramCuts(common::HistogramCuts const &cuts, dmlc::Stream *fo) {
  size_t bytes = 0;
  fo->Write(cuts.cut_values_.ConstHostVector());
  bytes += cuts.cut_values_.ConstHostSpan().size_bytes() + sizeof(uint64_t);
  fo->Write(cuts.cut_ptrs_.ConstHostVector());
  bytes += cuts.cut_ptrs_.ConstHostSpan().size_bytes() + sizeof(uint64_t);
  fo->Write(cuts.min_vals_.ConstHostVector());
  bytes += cuts.min_vals_.ConstHostSpan().size_bytes() + sizeof(uint64_t);
  return bytes;
}
}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_HISTOGRAM_CUT_FORMAT_H_
