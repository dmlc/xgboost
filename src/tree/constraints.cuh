/*!
 * Copyright 2019 XGBoost contributors
 *
 * \file Various constraints used in GPU_Hist.
 */
#ifndef XGBOOST_TREE_CONSTRAINTS_H_
#define XGBOOST_TREE_CONSTRAINTS_H_

#include <dmlc/json.h>

#include <cinttypes>
#include <vector>

#include "param.h"
#include "constraints.h"
#include "xgboost/span.h"
#include "../common/bitfield.h"
#include "../common/device_helpers.cuh"

namespace xgboost {
// Feature interaction constraints built for GPU Hist updater.
struct FeatureInteractionConstraintDevice {
 protected:
  // Whether interaction constraint is used.
  bool has_constraint_;
  // n interaction sets.
  size_t n_sets_;

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
   * d_sets_ptr_:                        |0, 1, 3, 4|
   */
  dh::device_vector<bst_feature_t> d_sets_;
  common::Span<bst_feature_t> s_sets_;
  dh::device_vector<size_t> d_sets_ptr_;
  common::Span<size_t> s_sets_ptr_;

  // Allowed features attached to each node, have n_nodes bitfields,
  // each of size n_features.
  std::vector<dh::device_vector<LBitField64::value_type>> node_constraints_storage_;
  std::vector<LBitField64> node_constraints_;
  common::Span<LBitField64> s_node_constraints_;

  // buffer storing return feature list from Query, of size n_features.
  dh::device_vector<bst_feature_t> result_buffer_;
  common::Span<bst_feature_t> s_result_buffer_;

  // Temp buffers, one bit for each possible feature.
  dh::device_vector<LBitField64::value_type> output_buffer_bits_storage_;
  LBitField64 output_buffer_bits_;
  dh::device_vector<LBitField64::value_type> input_buffer_bits_storage_;
  LBitField64 input_buffer_bits_;
  /*
   * Combined features from all interaction sets that one feature belongs to.
   * For an input with [[0, 1], [1, 2]], the feature 1 belongs to sets {0, 1}
   */
  dh::device_vector<LBitField64::value_type> d_feature_buffer_storage_;
  LBitField64 feature_buffer_;  // of Size n features.

  // Clear out all temp buffers except for `feature_buffer_', which is
  // handled in `Split'.
  void ClearBuffers();

 public:
  size_t Features() const;
  FeatureInteractionConstraintDevice() = default;
  void Configure(tree::TrainParam const& param, int32_t const n_features);
  FeatureInteractionConstraintDevice(tree::TrainParam const& param, int32_t const n_features);
  FeatureInteractionConstraintDevice(FeatureInteractionConstraintDevice const& that) = default;
  FeatureInteractionConstraintDevice(FeatureInteractionConstraintDevice&& that) = default;
  /*! \brief Reset before constructing a new tree. */
  void Reset();
  /*! \brief Return a list of features given node id */
  common::Span<bst_feature_t> QueryNode(int32_t nid);
  /*!
   * \brief Return a list of selected features from given feature_list and node id.
   *
   * \param feature_list A list of features
   * \param nid node id
   *
   * \return A list of features picked from `feature_list' that conform to constraints in
   * node.
   */
  common::Span<bst_feature_t> Query(common::Span<bst_feature_t> feature_list, int32_t nid);
  /*! \brief Apply split for node_id. */
  void Split(bst_node_t node_id, bst_feature_t feature_id, bst_node_t left_id, bst_node_t right_id);
};

}      // namespace xgboost
#endif  // XGBOOST_TREE_CONSTRAINTS_H_
