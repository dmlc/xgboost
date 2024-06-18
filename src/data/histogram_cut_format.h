/**
 * Copyright 2021-2024, XGBoost contributors
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
  bool has_cat{false};
  if (!fi->Read(&has_cat)) {
    return false;
  }
  decltype(cuts->MaxCategory()) max_cat{0};
  if (!fi->Read(&max_cat)) {
    return false;
  }
  cuts->SetCategorical(has_cat, max_cat);
  return true;
}

inline std::size_t WriteHistogramCuts(common::HistogramCuts const &cuts,
                                      common::AlignedFileWriteStream *fo) {
  std::size_t bytes = 0;
  bytes += common::WriteVec(fo, cuts.Values());
  bytes += common::WriteVec(fo, cuts.Ptrs());
  bytes += common::WriteVec(fo, cuts.MinValues());
  bytes += fo->Write(cuts.HasCategorical());
  bytes += fo->Write(cuts.MaxCategory());
  return bytes;
}
}  // namespace xgboost::data
#endif  // XGBOOST_DATA_HISTOGRAM_CUT_FORMAT_H_
