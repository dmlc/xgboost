/**
 * Copyright 2020-2024, XGBoost Contributors
 */

#include <algorithm>  // for max
#include <cstddef>    // for size_t
#include <cstdint>    // for uint32_t
#include <vector>     // for vector

#include "../../common/hist_util.h"  // for HistogramCuts
#include "feature_groups.cuh"

namespace xgboost::tree {
FeatureGroups::FeatureGroups(common::HistogramCuts const& cuts, bool is_dense, size_t shm_size)
    : max_group_bins{0} {
  // Only use a single feature group for sparse matrices.
  bool single_group = !is_dense;
  if (single_group) {
    InitSingle(cuts);
    return;
  }

  auto& feature_segments_h = feature_segments.HostVector();
  auto& bin_segments_h = bin_segments.HostVector();
  feature_segments_h.push_back(0);
  bin_segments_h.push_back(0);

  std::vector<std::uint32_t> const& cut_ptrs = cuts.Ptrs();
  // Maximum number of bins that can be placed into shared memory.
  std::size_t max_shmem_bins = shm_size / sizeof(GradientPairInt64);

  for (size_t i = 2; i < cut_ptrs.size(); ++i) {
    int last_start = bin_segments_h.back();
    // Push a new group whenever the size of required bin storage is greater than the
    // shared memory size.
    if (cut_ptrs[i] - last_start > max_shmem_bins) {
      feature_segments_h.push_back(i - 1);
      bin_segments_h.push_back(cut_ptrs[i - 1]);
      max_group_bins = std::max(max_group_bins, bin_segments_h.back() - last_start);
    }
  }
  feature_segments_h.push_back(cut_ptrs.size() - 1);
  bin_segments_h.push_back(cut_ptrs.back());
  max_group_bins =
      std::max(max_group_bins, bin_segments_h.back() - bin_segments_h[bin_segments_h.size() - 2]);
}

void FeatureGroups::InitSingle(common::HistogramCuts const& cuts) {
  auto& feature_segments_h = feature_segments.HostVector();
  feature_segments_h.push_back(0);
  feature_segments_h.push_back(cuts.Ptrs().size() - 1);

  auto& bin_segments_h = bin_segments.HostVector();
  bin_segments_h.push_back(0);
  bin_segments_h.push_back(cuts.TotalBins());

  max_group_bins = cuts.TotalBins();
}
}  // namespace xgboost::tree
