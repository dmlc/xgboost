#ifndef HISTOGRAM_H_
#define HISTOGRAM_H_

#include <xgboost/data.h>
#include "../common/quantile.h"
#include "../common/timer.h"

namespace xgboost {

using BinIdx = uint32_t;

// A CSC matrix representing quantile cuts
class CutMatrix {
 // private:
 public:
  std::vector<bst_float> cut_values_;
  std::vector<uint32_t> column_ptrs_;
  common::Monitor monitor_;

 public:
  CutMatrix();
  void Build(DMatrix* dmat, uint32_t const max_num_bins);
  BinIdx SearchBin(float value, uint32_t column_id) const;
};

// records index of each data entry. CSC
class HistogramIndices {
 public:
  std::vector<uint32_t> column_ptrs_;
  std::vector<uint32_t> indices_;

 public:
  HistogramIndices();
  void Build(DMatrix* dmat, CutMatrix const& cut, uint32_t const max_num_bins);
};

}      // namespace xgboost
#endif  // HISTOGRAM_H_
