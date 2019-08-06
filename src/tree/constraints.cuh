/*!
 * Copyright 2019 XGBoost contributors
 */
#ifndef XGBOOST_TREE_CONSTRAINTS_H_
#define XGBOOST_TREE_CONSTRAINTS_H_

#include <dmlc/json.h>

#include <cinttypes>
#include <vector>

#include "param.h"
#include "../common/span.h"
#include "../common/bitfield.cuh"
#include "../common/device_helpers.cuh"

namespace xgboost {

// Feature interaction constraints built for GPU Hist updater.
struct FeatureInteractionConstraint {
 protected:
  // Whether interaction constraint is used.
  bool has_constraint_;
  // n interaction sets.
  int32_t n_sets_;

  // The parsed feature interaction constraints as CSR.
  dh::device_vector<int32_t> d_fconstraints_;
  common::Span<int32_t> s_fconstraints_;
  dh::device_vector<int32_t> d_fconstraints_ptr_;
  common::Span<int32_t> s_fconstraints_ptr_;
  /* Interaction sets for each feature as CSR.  For an input like:
   * [[0, 1], [1, 2]], this will have values:
   *
   * fid:                                |0 | 1  | 2|
   * sets a feature belongs to(d_sets_): |0 |0, 1| 1|
   *
   * d_sets_ptr_:                        |0, 1, 3, 4|
   */
  dh::device_vector<int32_t> d_sets_;
  common::Span<int32_t> s_sets_;
  dh::device_vector<int32_t> d_sets_ptr_;
  common::Span<int32_t> s_sets_ptr_;

  // Allowed features attached to each node, have n_nodes bitfields,
  // each of size n_features.
  std::vector<dh::device_vector<BitField::value_type>> node_constraints_storage_;
  std::vector<BitField> node_constraints_;
  common::Span<BitField> s_node_constraints_;

  // buffer storing return feature list from Query, of size n_features.
  dh::device_vector<int32_t> result_buffer_;
  common::Span<int32_t> s_result_buffer_;

  // Temp buffers, one bit for each possible feature.
  dh::device_vector<BitField::value_type> output_buffer_bits_storage_;
  BitField output_buffer_bits_;
  dh::device_vector<BitField::value_type> input_buffer_bits_storage_;
  BitField input_buffer_bits_;
  /*
   * Combined features from all interaction sets that one feature belongs to.
   * For an input with [[0, 1], [1, 2]], the feature 1 belongs to sets {0, 1}
   */
  dh::device_vector<BitField::value_type> d_feature_buffer_storage_;
  BitField feature_buffer_;  // of Size n features.

  // Clear out all temp buffers except for `feature_buffer_', which is
  // handled in `Split'.
  void ClearBuffers();

 public:
  size_t Features() const;
  FeatureInteractionConstraint() = default;
  void Configure(tree::TrainParam const& param, int32_t const n_features);
  FeatureInteractionConstraint(tree::TrainParam const& param, int32_t const n_features);
  FeatureInteractionConstraint(FeatureInteractionConstraint const& that) = default;
  FeatureInteractionConstraint(FeatureInteractionConstraint&& that) = default;
  /*! \brief Reset before constructing a new tree. */
  void Reset();
  /*! \brief Return a list of features given node id */
  common::Span<int32_t> QueryNode(int32_t nid);
  /*!
   * \brief Return a list of selected features from given feature_list and node id.
   *
   * \param feature_list A list of features
   * \param nid node id
   *
   * \return A list of features picked from `feature_list' that conform to constraints in
   * node.
   */
  common::Span<int32_t> Query(common::Span<int32_t> feature_list, int32_t nid);
  /*! \brief Apply split for node_id. */
  void Split(int32_t node_id, int32_t feature_id, int32_t left_id, int32_t right_id);
};

}      // namespace xgboost
#endif  // XGBOOST_TREE_CONSTRAINTS_H_
