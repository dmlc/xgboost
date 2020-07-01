/*!
 * Copyright 2020 by XGBoost Contributors
 */

#include <xgboost/base.h>

#include "feature_groups.cuh"

#include "../../common/device_helpers.cuh"
#include "../../common/hist_util.h"

namespace xgboost {
namespace tree {

template <typename GradientSumT>
void FeatureGroups::Init(const common::HistogramCuts& cuts, bool is_dense,
                         int shm_size) {
  std::vector<int>& feature_segments_h = feature_segments.HostVector();
  std::vector<int>& bin_segments_h = bin_segments.HostVector();
  feature_segments_h.push_back(0);
  bin_segments_h.push_back(0);

  // Don't use feature groups for sparse matrices.
  bool single_group = !is_dense;
  if (single_group) {
    feature_segments_h.push_back(cuts.Ptrs().size() - 1);
    bin_segments_h.push_back(cuts.TotalBins());
    max_group_bins = cuts.TotalBins();
    return;
  }

  const std::vector<uint32_t>& cut_ptrs = cuts.Ptrs();
  int max_shmem_bins = shm_size / sizeof(GradientSumT);
  max_group_bins = 0;

  for (size_t i = 2; i < cut_ptrs.size(); ++i) {
    int last_start = bin_segments_h.back();
    if (cut_ptrs[i] - last_start > max_shmem_bins) {
      feature_segments_h.push_back(i - 1);
      bin_segments_h.push_back(cut_ptrs[i - 1]);
      max_group_bins = std::max(max_group_bins,
                                bin_segments_h.back() - last_start);
    }
  }
  feature_segments_h.push_back(cut_ptrs.size() - 1);
  bin_segments_h.push_back(cut_ptrs.back());
  max_group_bins = std::max(max_group_bins,
                            bin_segments_h.back() -
                            bin_segments_h[bin_segments_h.size() - 2]);
}

template void FeatureGroups::Init<GradientPair>(
    const common::HistogramCuts& cuts, bool is_dense, int shm_size);

template void FeatureGroups::Init<GradientPairPrecise>(
    const common::HistogramCuts& cuts, bool is_dense, int shm_size);

}  // namespace tree
}  // namespace xgboost
