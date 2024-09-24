/**
 * Copyright 2020-2024, XGBoost Contributors
 */
#ifndef FEATURE_GROUPS_CUH_
#define FEATURE_GROUPS_CUH_

#include <xgboost/host_device_vector.h>
#include <xgboost/span.h>

namespace xgboost {

// Forward declarations.
namespace common {
class HistogramCuts;
}  // namespace common

namespace tree {

/**
 * @brief FeatureGroup is a single group of features.
 *
 * It is defined by a range of consecutive feature indices, and also contains a range of
 * all bin indices associated with those features.
 */
struct FeatureGroup {
  XGBOOST_DEVICE FeatureGroup(bst_feature_t start_feature, bst_feature_t n_features,
                              bst_bin_t start_bin, bst_bin_t num_bins)
      : start_feature{start_feature},
        num_features{n_features},
        start_bin{start_bin},
        num_bins{num_bins} {}
  /** The first feature of the group. */
  bst_feature_t start_feature;
  /** The number of features in the group. */
  bst_feature_t num_features;
  /** The first bin in the group. */
  bst_bin_t start_bin;
  /** The number of bins in the group. */
  bst_bin_t num_bins;
};

/** @brief FeatureGroupsAccessor is a non-owning accessor for FeatureGroups. */
struct FeatureGroupsAccessor {
  FeatureGroupsAccessor(common::Span<const bst_feature_t> feature_segments,
                        common::Span<const bst_bin_t> bin_segments, bst_bin_t max_group_bins)
      : feature_segments{feature_segments},
        bin_segments{bin_segments},
        max_group_bins{max_group_bins} {}

  common::Span<const bst_feature_t> feature_segments;
  common::Span<const int> bin_segments;
  bst_bin_t max_group_bins;

  /** @brief Gets the number of feature groups. */
  XGBOOST_DEVICE int NumGroups() const { return feature_segments.size() - 1; }

  /** @brief Gets the information about a feature group with index i. */
  XGBOOST_DEVICE FeatureGroup operator[](bst_feature_t i) const {
    return {feature_segments[i], feature_segments[i + 1] - feature_segments[i], bin_segments[i],
            bin_segments[i + 1] - bin_segments[i]};
  }
};

/**
 * @brief FeatureGroups contains information that defines a split of features
 *   into groups. Bins of a single feature group typically fit into shared
 *   memory, so the histogram for the features of a single group can be computed
 *   faster.
 *
 * @note Known limitations:
 *
 *  - splitting features into groups currently works only for dense matrices,
 *    where it is easy to get a feature value in a row by its index; for sparse
 *    matrices, the structure contains only a single group containing all
 *    features;
 *
 *  - if a single feature requires more bins than fit into shared memory, the
 *    histogram is computed in global memory even if there are multiple feature
 *    groups; note that this is unlikely to occur in practice, as the default
 *    number of bins per feature is 256, whereas a thread block with 48 KiB
 *    shared memory can contain 3072 bins if each gradient sum component is a
 *     64-bit floating-point value (double)
*/
struct FeatureGroups {
  /** Group cuts for features. Size equals to (number of groups + 1). */
  HostDeviceVector<bst_feature_t> feature_segments;
  /** Group cuts for bins. Size equals to (number of groups + 1)  */
  HostDeviceVector<int> bin_segments;
  /** Maximum number of bins in a group. Useful to compute the amount of dynamic
      shared memory when launching a kernel. */
  int max_group_bins;

  /**
   * @brief Creates feature groups by splitting features into groups.
   *
   * @param cuts Histogram cuts that given the number of bins per feature.
   * @param is_dense Whether the data matrix is dense.
   * @param shm_size Available size of shared memory per thread block (in bytes) used to
   *  compute feature groups.
   */
  FeatureGroups(common::HistogramCuts const& cuts, bool is_dense, size_t shm_size);

  /**
   * @brief Creates a single feature group containing all features and bins.
   *
   * @notes This is used as a fallback for sparse matrices, and is also useful for
   *        testing.
   */
  explicit FeatureGroups(const common::HistogramCuts& cuts) { this->InitSingle(cuts); }

  [[nodiscard]] FeatureGroupsAccessor DeviceAccessor(DeviceOrd device) const {
    feature_segments.SetDevice(device);
    bin_segments.SetDevice(device);
    return {feature_segments.ConstDeviceSpan(), bin_segments.ConstDeviceSpan(), max_group_bins};
  }

private:
  void InitSingle(const common::HistogramCuts& cuts);
};

}  // namespace tree
}  // namespace xgboost

#endif  // FEATURE_GROUPS_CUH_
