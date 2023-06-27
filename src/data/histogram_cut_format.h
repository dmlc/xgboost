/**
 * Copyright 2021-2023, XGBoost contributors
 */
#ifndef XGBOOST_DATA_HISTOGRAM_CUT_FORMAT_H_
#define XGBOOST_DATA_HISTOGRAM_CUT_FORMAT_H_

#include <dmlc/io.h>  // for Stream

#include <cstddef>  // for size_t

#include "../common/hist_util.h"          // for HistogramCuts
#include "../common/io.h"                 // for AlignedResourceReadStream, AlignedFileWriteStream
#include "../common/ref_resource_view.h"  // for WriteVec, ReadVec

namespace xgboost::data {
inline bool ReadHistogramCuts(common::HistogramCuts *cuts, common::AlignedResourceReadStream *fi) {
  if (!common::ReadVec(fi, &cuts->cut_values_.HostVector())) {
    return false;
  }
  if (!common::ReadVec(fi, &cuts->cut_ptrs_.HostVector())) {
    return false;
  }
  if (!common::ReadVec(fi, &cuts->min_vals_.HostVector())) {
    return false;
  }
  return true;
}

inline std::size_t WriteHistogramCuts(common::HistogramCuts const &cuts,
                                      common::AlignedFileWriteStream *fo) {
  std::size_t bytes = 0;
  bytes += common::WriteVec(fo, cuts.Values());
  bytes += common::WriteVec(fo, cuts.Ptrs());
  bytes += common::WriteVec(fo, cuts.MinValues());
  return bytes;
}
}  // namespace xgboost::data
#endif  // XGBOOST_DATA_HISTOGRAM_CUT_FORMAT_H_
