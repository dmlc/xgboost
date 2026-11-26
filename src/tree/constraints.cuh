/**
 * Copyright 2019-2026, XGBoost contributors
 *
 * \file Various constraints used in GPU_Hist.
 */
#ifndef XGBOOST_TREE_CONSTRAINTS_CUH_
#define XGBOOST_TREE_CONSTRAINTS_CUH_

#include <cstddef>  // for size_t
#include <vector>   // for vector

#include "../common/bitfield.h"         // for LBitField64
#include "../common/device_vector.cuh"  // for CachingDeviceUVector, device_vector
#include "param.h"                      // for TrainParam
#include "xgboost/span.h"               // for Span

namespace xgboost {
// Feature interaction constraints built for GPU Hist updater.
struct FeatureInteractionConstraintDevice {
 protected:
  // Whether interaction constraint is used.
  bool has_constraint_;
  // n interaction sets.
  std::size_t n_sets_;

  // The parsed feature interaction constraints as CSR.
  dh::device_vector<bst_feature_t> d_fconstraints_;
  common::Span<bst_feature_t> s_fconstraints_;
  dh::device_vector<size_t> d_fconstraints_ptr_;
  common::Span<size_t> s_fconstraints_ptr_;
  /* Interaction sets for each feature as CSR.  For an input like:
   * [[0, 1], [1, 2]], this will have values:
   *
   * fid:                                |0 | 1  | 2|
   * sets a feature belongs to(d_sets_): |0 |0, 1| 1|
   *
   * h_sets_ptr_:                        |0, 1, 3, 4|
   */
  dh::device_vector<bst_feature_t> d_sets_;
  common::Span<bst_feature_t> s_sets_;
  // Host offsets into d_sets_ for looking up the groups containing a split feature.
  std::vector<std::size_t> h_sets_ptr_;

  // Allowed features attached to each node, have n_nodes bitfields,
  // each of size n_features.
  std::vector<dh::device_vector<LBitField64::value_type>> node_constraints_storage_;
  std::vector<LBitField64> node_constraints_;
  common::Span<LBitField64> s_node_constraints_;

  // Reused across batches. Each node gets a disjoint slice for its filtered features.
  dh::CachingDeviceUVector<bst_feature_t> result_buffer_;
  // One bit per interaction group. A set bit means the group contains the entire path.
  // Logical size is n_groups x n_nodes bits.
  dh::CachingDeviceUVector<LBitField64::value_type> node_groups_;

 public:
  size_t Features() const;
  FeatureInteractionConstraintDevice() = default;
  void Configure(tree::TrainParam const& param, int32_t const n_features);
  FeatureInteractionConstraintDevice(tree::TrainParam const& param, int32_t const n_features);
  FeatureInteractionConstraintDevice(FeatureInteractionConstraintDevice const& that) = delete;
  FeatureInteractionConstraintDevice(FeatureInteractionConstraintDevice&& that) = default;
  /*! \brief Reset before constructing a new tree. */
  void Reset(Context const* ctx);
  /** @brief Get output storage for all queries in a batch. Reuse it after evaluation finishes. */
  common::Span<bst_feature_t> PrepareBatch(std::size_t n_sampled);
  /** @brief Filter sorted, unique sampled features into the node's output slice.
   *  @return The sampled features for an unrestricted node, or the populated prefix of out.
   */
  common::Span<bst_feature_t const> Query(Context const* ctx,
                                          common::Span<bst_feature_t const> feature_list,
                                          bst_node_t nidx, common::Span<bst_feature_t> out) const;
  /*! \brief Apply split for node_id. */
  void Split(Context const* ctx, bst_node_t node_id, bst_feature_t feature_id, bst_node_t left_id,
             bst_node_t right_id);
};

}  // namespace xgboost
#endif  // XGBOOST_TREE_CONSTRAINTS_CUH_
